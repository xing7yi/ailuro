#include "PlotPostprocessor.h"
#include "ExecutablePath.h"
#include <fstream>
#include <sstream>

registerMooseObject("AiluroApp", PlotPostprocessor);

InputParameters
PlotPostprocessor::validParams()
{
  InputParameters params = GeneralPostprocessor::validParams();

  params.addClassDescription("Postprocessor that generates real-time plots of specified variables");

  params.addRequiredParam<PostprocessorName>("x_variable", "Postprocessor name for x-axis");
  params.addRequiredParam<PostprocessorName>("y_variable", "Postprocessor name for y-axis");
  params.addParam<std::string>("x_label", "", "Label for the x-axis");
  params.addParam<std::string>("y_label", "", "Label for the y-axis");
  params.addParam<std::string>("plot_title", "", "Title for the plot");
  params.addParam<std::string>("style", "", "Plot style (e.g., line, scatter)");
  params.addParam<bool>("real_time_plot", false, "Generate plot at each time step");
  params.addParam<unsigned int>("plot_frequency", 1, "Plot every N time steps");

  // Add EXEC_FINAL to execute_on to ensure final plot is generated
  ExecFlagEnum & exec_enum = params.set<ExecFlagEnum>("execute_on", true);
  exec_enum.addAvailableFlags(EXEC_FINAL);
  exec_enum += EXEC_FINAL;

  return params;
}

PlotPostprocessor::PlotPostprocessor(const InputParameters & parameters)
  : GeneralPostprocessor(parameters),
    _x_variable(getParam<PostprocessorName>("x_variable")),
    _y_variable(getParam<PostprocessorName>("y_variable")),
    _x_label(getParam<std::string>("x_label").empty() ? _x_variable
                                                      : getParam<std::string>("x_label")),
    _y_label(getParam<std::string>("y_label").empty() ? _y_variable
                                                      : getParam<std::string>("y_label")),
    _plot_title(getParam<std::string>("plot_title")),
    _style(getParam<std::string>("style")),
    _real_time_plot(getParam<bool>("real_time_plot")),
    _plot_frequency(getParam<unsigned int>("plot_frequency")),
    _time_step_counter(0),
    _python_pipe(nullptr),
    _python_initialized(false)
{
  // Generate output file name
  _output_file = _app.getOutputFileBase() + "_" + name() + ".png";
}

void
PlotPostprocessor::initialize()
{
  if (processor_id() != 0)
    return;

  // Update output filename based on current file_base
  std::string current_output_file = _app.getOutputFileBase() + "_" + name() + ".png";

  // Detect new sub-app: output filename has changed
  if (current_output_file != _output_file)
  {
    // Close existing Python process if any
    closePythonProcess();  
    // Update output filename
    _output_file = current_output_file;

  }

  // Initialize Python process if needed
  if (!_python_initialized)
    initializePythonProcess();
}

void
PlotPostprocessor::execute()
{
  if (processor_id() != 0)
    return;

  // Regular execution: accumulate data
  _time_step_counter++;

  Real x_val = getPostprocessorValueByName(_x_variable);
  Real y_val = getPostprocessorValueByName(_y_variable);
  _x_data.push_back(x_val);
  _y_data.push_back(y_val);

  // Update plot at specified frequency if real_time_plot is enabled
  bool reach_final = (_fe_problem.getCurrentExecuteOnFlag() == EXEC_FINAL);
  _console << "PlotPostprocessor: reach_final=" << reach_final << ", _real_time_plot=" << _real_time_plot << ", _time_step_counter=" << _time_step_counter << ", _plot_frequency=" << _plot_frequency << std::endl;
  if ((_real_time_plot && (_time_step_counter % _plot_frequency == 0)) || reach_final)
  {
    _console << "PlotPostprocessor: Updating plot at step " << _time_step_counter << std::endl;
    updatePlotData(_x_data, _y_data);
  }
}

void
PlotPostprocessor::finalize()
{
}

PlotPostprocessor::~PlotPostprocessor()
{
  if (processor_id() != 0)
    return;

  closePythonProcess();
}

void
PlotPostprocessor::initializePythonProcess()
{
  if (_python_initialized)
  return;

  _console << "PlotPostprocessor: Initializing persistent Python plotting process..." << std::endl;

  // Search for Ailuro's standalone Python plotting tool
  std::vector<std::string> search_paths;

  // Use getExecutablePath() to get full path, not just the name
  std::string exe_path = Moose::getExecutablePath();

  if (!exe_path.empty())
  {
    // obtain directory of the executable
    size_t last_slash = exe_path.find_last_of("/\\");
    if (last_slash != std::string::npos)
    {
      std::string exe_dir = exe_path.substr(0, last_slash);
      search_paths.push_back(exe_dir + "/python/realtime_plotter.py");
    }
  }

  std::string plotter_script;
  for (const auto & path : search_paths)
  {
    std::ifstream test(path);
    if (test.good())
    {
      plotter_script = path;
      break;
    }
  }

  if (plotter_script.empty())
  {
    mooseError("PlotPostprocessor: Cannot find realtime_plotter.py. "
               "Set AILURO_DIR or run from build directory.");
  }

  std::stringstream cmd_builder;
  cmd_builder << "python3 -u " << plotter_script;
  cmd_builder << " --output \"" << _output_file << "\"";
  if (_x_label != "")
    cmd_builder << " --xlabel \"" << _x_label << "\"";
  if (_y_label != "")
    cmd_builder << " --ylabel \"" << _y_label << "\"";
  if (_plot_title != "")
    cmd_builder << " --title \"" << _plot_title << "\"";
  if (_style != "")
    cmd_builder << " --style \"" << _style << "\"";
  cmd_builder << " 2>&1";

  std::string cmd = cmd_builder.str();

  // start a separate Python plotting process (receiving data via stdin)
  _python_pipe = popen(cmd.c_str(), "w");

  if (_python_pipe == nullptr)
  {
    mooseError("PlotPostprocessor: Failed to start Python plotting process");
  }

  _python_initialized = true;
}

void
PlotPostprocessor::closePythonProcess()
{
  if (!_python_initialized || _python_pipe == nullptr)
    return;

  _console << "PlotPostprocessor: Closing Python plotting process..." << std::endl;

  // send close signal
  fprintf(_python_pipe, "CLOSE\n");
  fflush(_python_pipe);

  // close pipe
  pclose(_python_pipe);
  _python_pipe = nullptr;
  _python_initialized = false;

  // Clear data
  _x_data.clear();
  _y_data.clear();
  _time_step_counter = 0;
}

void
PlotPostprocessor::updatePlotData(const std::vector<Real> & x_data,
                                  const std::vector<Real> & y_data)
{
  if (!_python_initialized || _python_pipe == nullptr)
  {
    _console << "WARNING: Python process not initialized, skipping plot update" << std::endl;
    return;
  }

  if (x_data.size() != y_data.size())
  {
    _console << "ERROR: x_data and y_data size mismatch" << std::endl;
    return;
  }

  // send data: use DATA: prefix + number of points + all x,y pairs
  fprintf(_python_pipe, "DATA:%zu\n", x_data.size());
  for (size_t i = 0; i < x_data.size(); ++i)
  {
    fprintf(_python_pipe, "%.16e,%.16e\n", x_data[i], y_data[i]);
  }
  // Send END marker
  fprintf(_python_pipe, "END\n");

  int flush_result = fflush(_python_pipe);
  if (flush_result != 0)
  {
    _console << "ERROR: Failed to flush Python pipe" << std::endl;
    return;
  }
}

PostprocessorValue
PlotPostprocessor::getValue() const
{
  return _time_step_counter;
}