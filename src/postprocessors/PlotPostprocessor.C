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
    _output_python_script(getParam<bool>("output_python_script")),
    _python_pipe(nullptr),
    _python_initialized(false)
{
}

void
PlotPostprocessor::initialize()
{
  // 仅在实时绘图模式下初始化持久 Python 进程
  if (_real_time_plot && processor_id() == 0 && !_python_initialized)
  {
    initializePythonProcess();
  }
}

void
PlotPostprocessor::finalize()
{
  // finalize() 在每个时间步都会被调用
  // 对于批量模式，我们在析构函数中生成图表
  // 对于实时模式，Python 进程会一直运行直到析构
}

PlotPostprocessor::~PlotPostprocessor()
{
  // 析构函数：确保 Python 进程被正确关闭
  if (_real_time_plot && processor_id() == 0)
  {
    closePythonProcess();
  }
  // 批量模式：在析构时一次性生成图表
  else if (!_real_time_plot && processor_id() == 0)
  {
    generatePlot();
  }
}

void
PlotPostprocessor::execute()
{
  _time_step_counter++;

  Real x_val = getPostprocessorValueByName(_x_variable);
  Real y_val = getPostprocessorValueByName(_pp_names[0]);

  // 实时绘图模式：通过管道更新数据（按频率）
  if (_real_time_plot && _python_initialized && (_time_step_counter % _plot_frequency == 0))
  {
    if (processor_id() == 0)
    {
      _console << "PlotPostprocessor: Updating plot at step " << _time_step_counter 
               << " with x=" << x_val << ", y=" << y_val << std::endl;
      updatePlotData(x_val, y_val);
    }
  }
  
  // 批量模式：缓存所有数据
  if (!_real_time_plot)
  {
    _x_data.push_back(x_val);
    _y_data.push_back(y_val);
  }
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

void
PlotPostprocessor::initializePythonProcess()
{
  if (_python_initialized)
    return;

  _console << "PlotPostprocessor: Initializing persistent Python plotting process..." << std::endl;

  // 创建交互式 Python 脚本
  std::string interactive_script = _python_script_path + ".interactive.py";
  std::ofstream script(interactive_script);

  script << "#!/usr/bin/env python3\n";
  script << "import sys\n";
  script << "import numpy as np\n";
  script << "import matplotlib\n";
  script << "matplotlib.use('Agg')  # Non-interactive backend\n";
  script << "import matplotlib.pyplot as plt\n\n";

  script << "# Initialize plot\n";
  script << "fig, ax = plt.subplots(figsize=(5, 4))\n";
  script << "plot_line, = ax.plot([], [], 'bo-', markersize=5, markerfacecolor='none')\n";
  script << "ax.set_xlabel('" << _x_label << "')\n";
  script << "ax.set_ylabel('" << _y_label << "')\n";
  script << "ax.set_title('" << _plot_title << "')\n";
  script << "ax.grid(True, alpha=0.3)\n";
  script << "\n";
  script << "x_data = []\n";
  script << "y_data = []\n";
  script << "\n";
  script << "# Signal ready and start reading\n";
  script << "print('========== PYTHON PLOT PROCESS READY ==========', flush=True)\n";
  script << "sys.stderr.write('Python plotting initialized\\n')\n";
  script << "sys.stderr.flush()\n";
  script << "\n";
  script << "# Read data from stdin\n";
  script << "for input_line in sys.stdin:\n";
  script << "    input_line = input_line.strip()\n";
  script << "    if input_line == 'CLOSE':\n";
  script << "        break\n";
  script << "    \n";
  script << "    # Parse x,y data\n";
  script << "    try:\n";
  script << "        x_str, y_str = input_line.split(',')\n";
  script << "        x_val = float(x_str)\n";
  script << "        y_val = float(y_str)\n";
  script << "        \n";
  script << "        x_data.append(x_val)\n";
  script << "        y_data.append(y_val)\n";
  script << "        \n";
  script << "        # Update plot\n";
  script << "        plot_line.set_data(x_data, y_data)\n";
  script << "        ax.relim()\n";
  script << "        ax.autoscale_view()\n";
  script << "        \n";
  script << "        # Save figure\n";
  script << "        plt.tight_layout()\n";
  script << "        fig.savefig('" << _output_file << "', dpi=300)\n";
  script << "        \n";
  script << "    except Exception as e:\n";
  script << "        print(f'ERROR: {e}', flush=True)\n";
  script << "\n";
  script << "print('CLOSED', flush=True)\n";

  script.close();

  // 启动 Python 进程并建立管道（后台运行，重定向 stderr）
  std::string cmd = "python3 -u " + interactive_script + " 2>&1";
  _python_pipe = popen(cmd.c_str(), "w");

  if (_python_pipe == nullptr)
  {
    mooseError("PlotPostprocessor: Failed to start Python plotting process");
  }

  _python_initialized = true;
  _console << "PlotPostprocessor: Plotting pipeline ready. Updates will be sent at frequency=" 
           << _plot_frequency << std::endl;
}

void
PlotPostprocessor::closePythonProcess()
{
  if (!_python_initialized || _python_pipe == nullptr)
    return;

  _console << "PlotPostprocessor: Closing Python plotting process..." << std::endl;

  // 发送关闭信号
  fprintf(_python_pipe, "CLOSE\n");
  fflush(_python_pipe);

  // 关闭管道
  pclose(_python_pipe);
  _python_pipe = nullptr;
  _python_initialized = false;

  _console << "PlotPostprocessor: Plot saved to " << _output_file << std::endl;
}

void
PlotPostprocessor::updatePlotData(Real x_val, Real y_val)
{
  if (!_python_initialized || _python_pipe == nullptr)
  {
    _console << "WARNING: Python process not initialized, skipping plot update" << std::endl;
    return;
  }

  // 通过管道发送数据到 Python 进程
  int result = fprintf(_python_pipe, "%.16e,%.16e\n", x_val, y_val);
  if (result < 0)
  {
    _console << "ERROR: Failed to write to Python pipe" << std::endl;
    return;
  }
  
  int flush_result = fflush(_python_pipe);
  if (flush_result != 0)
  {
    _console << "ERROR: Failed to flush Python pipe" << std::endl;
    return;
  }
  
  _console << "  -> Data sent to Python process successfully" << std::endl;
}