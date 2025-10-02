#pragma once

#include "GeneralPostprocessor.h"
#include "FunctionParserUtils.h"

class PlotPostprocessor : public GeneralPostprocessor
{
public:
  static InputParameters validParams();

  PlotPostprocessor(const InputParameters & parameters);
  virtual ~PlotPostprocessor();

  virtual void initialize() override;
  virtual void execute() override;
  virtual PostprocessorValue getValue() const override;
  virtual void finalize() override;

protected:
  PostprocessorName _x_variable;
  PostprocessorName _y_variable;

  std::string _x_label;

  std::string _y_label;

  std::string _plot_title;

  std::string _output_file;

  bool _real_time_plot;

  unsigned int _plot_frequency;

  unsigned int _time_step_counter;

  std::vector<Real> _y_data;
  std::vector<Real> _x_data;

private:
  FILE * _python_pipe;

  bool _python_initialized;

  void initializePythonProcess();

  void closePythonProcess();

  void updatePlotData(const std::vector<Real> & x_data, const std::vector<Real> & y_data);
};