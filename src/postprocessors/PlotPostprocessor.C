#include "PlotPostprocessor.h"
#include <fstream>
#include <sstream>

registerMooseObject("AiluroApp", PlotPostprocessor);

InputParameters
PlotPostprocessor::validParams()
{
  InputParameters params = GeneralPostprocessor::validParams();

  params.addClassDescription("Postprocessor that generates real-time plots of specified variables");

  params.addRequiredParam<std::vector<PostprocessorName>>("pp_names",
                                                          "List of postprocessor names to plot");

  params.addRequiredParam<PostprocessorName>("x_variable",
                                             "Postprocessor name for x-axis (typically time)");

  params.addParam<std::string>("x_label", "", "Label for the x-axis");                                             
  params.addParam<std::string>("y_label", "", "Label for the y-axis");
  params.addParam<std::string>("plot_title", "Convergence Analysis", "Title for the plot");

  params.addParam<std::string>("output_file", "ailuro_plot.png", "Output image file name");

  params.addParam<bool>("real_time_plot", false, "Generate plot at each time step");

  params.addParam<unsigned int>("plot_frequency", 1, "Plot every N time steps");

  params.addParam<bool>("output_python_script", true, "Output Python script for plotting");

  params.addParam<std::string>("python_script_path", "ailuro_plot.py", "Path for Python script");

  return params;
}

PlotPostprocessor::PlotPostprocessor(const InputParameters & parameters)
  : GeneralPostprocessor(parameters),
    _pp_names(getParam<std::vector<PostprocessorName>>("pp_names")),
    _x_variable(getParam<PostprocessorName>("x_variable")),
    _x_label(getParam<std::string>("x_label")),
    _y_label(getParam<std::string>("y_label")),
    _plot_title(getParam<std::string>("plot_title")),
    _output_file(getParam<std::string>("output_file")),
    _real_time_plot(getParam<bool>("real_time_plot")),
    _plot_frequency(getParam<unsigned int>("plot_frequency")),
    _time_step_counter(0),
    _python_script_path(getParam<std::string>("python_script_path")),
    _output_python_script(getParam<bool>("output_python_script"))
{
}

void
PlotPostprocessor::initialize()
{
}

void
PlotPostprocessor::finalize()
{
  // Only generate plot if:
  // 1. real_time_plot is true (handled in execute()), OR
  // 2. We are at EXEC_FINAL (end of simulation)
if ((_real_time_plot && (_time_step_counter % _plot_frequency == 0)) ||
        (!_real_time_plot && _fe_problem.getCurrentExecuteOnFlag() == EXEC_FINAL))
{
    if (processor_id() == 0)
        generatePlot();
}
}

void
PlotPostprocessor::execute()
{
  _time_step_counter++;

  Real x_val = getPostprocessorValueByName(_x_variable);
  _x_data.push_back(x_val);

  Real y_val = getPostprocessorValueByName(_pp_names[0]);
  _y_data.push_back(y_val);
}

PostprocessorValue
PlotPostprocessor::getValue() const
{
  return _time_step_counter;
}

void
PlotPostprocessor::generatePlot()
{
  if (_x_data.empty())
  {
    _console << "PlotPostprocessor: No data to plot" << std::endl;
    return;
  }

  _console << "PlotPostprocessor: Generating plot with " << _x_data.size() << " data points"
           << std::endl;

  // Generate Python plotting script
  std::ofstream script(_python_script_path);

  script << "#!/usr/bin/env python3\n";
  script << "import numpy as np\n";
  script << "import matplotlib.pyplot as plt\n\n";

  // Write x data
  script << "# Data from Ailuro simulation\n";
  script << "x_data = np.array([";
  for (size_t i = 0; i < _x_data.size(); ++i)
  {
    if (i > 0)
      script << ", ";
    script << _x_data[i];
  }
  script << "])\n\n";

  // Write y data (currently only supporting first pp_name)
  script << "y_data = np.array([";
  for (size_t i = 0; i < _y_data.size(); ++i)
  {
    if (i > 0)
      script << ", ";
    script << _y_data[i];
  }
  script << "])\n\n";

  // Generate plot
  script << "# Create plot\n";
  script << "plt.figure(figsize=(5, 4))\n";
  script << "plt.plot(x_data, y_data, 'bo-', markersize=5 ,markerfacecolor='none')\n";
  script << "plt.xlabel('" << _x_label << "')\n";
  script << "plt.ylabel('" << _y_label << "')\n";
  script << "plt.title('" << _plot_title << "')\n";
  script << "plt.grid(True, alpha=0.3)\n";
  script << "plt.tight_layout()\n";
  script << "plt.savefig('" << _output_file << "', dpi=300)\n";
  script << "print('Plot saved to " << _output_file << "')\n";

  script.close();

  // Execute Python script if requested
  if (_output_python_script)
  {
    std::string cmd = "python3 " + _python_script_path;
    int result = system(cmd.c_str());
    if (result == 0)
      _console << "Plot generated successfully: " << _output_file << std::endl;
    else
      _console << "Warning: Failed to execute plotting script" << std::endl;
  }
}