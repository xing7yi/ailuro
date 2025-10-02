#pragma once

#include "GeneralPostprocessor.h"
#include "FunctionParserUtils.h"

class PlotPostprocessor : public GeneralPostprocessor
{
public:
  static InputParameters validParams();

  PlotPostprocessor(const InputParameters & parameters);

  virtual void initialize() override;
  virtual void execute() override;
  virtual PostprocessorValue getValue() const override;
  virtual void finalize() override;

protected:
  /// 要绘制的 postprocessor 名称列表
  std::vector<PostprocessorName> _pp_names;

  /// x 轴变量名（通常是时间或位移）
  PostprocessorName _x_variable;

  /// x axis label
  std::string _x_label;

  /// y axis label
  std::string _y_label;

  /// 图表标题
  std::string _plot_title;

  /// 输出文件名
  std::string _output_file;

  /// 是否实时更新图表
  bool _real_time_plot;

  /// 绘图频率（每多少个时间步绘制一次）
  unsigned int _plot_frequency;

  /// 时间步计数器
  unsigned int _time_step_counter;

  /// 存储历史数据
  std::vector<Real> _y_data;
  std::vector<Real> _x_data;

private:
  /// 生成图表的函数
  void generatePlot();

  /// Python 脚本路径（用于调用 matplotlib）
  std::string _python_script_path;

  /// 是否输出 Python 脚本
  bool _output_python_script;
};