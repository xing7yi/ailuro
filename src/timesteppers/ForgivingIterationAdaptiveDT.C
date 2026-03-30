// ForgivingIterationAdaptiveDT.C
#include "ForgivingIterationAdaptiveDT.h"

registerMooseObject("AiluroApp", ForgivingIterationAdaptiveDT);

InputParameters
ForgivingIterationAdaptiveDT::validParams()
{
  InputParameters params = IterationAdaptiveDT::validParams();
  params.addClassDescription(
      "Iteration adaptive timestepper that allows continuing when solve fails at dtmin");
  return params;
}

ForgivingIterationAdaptiveDT::ForgivingIterationAdaptiveDT(const InputParameters & parameters)
  : IterationAdaptiveDT(parameters)
{
}

Real
ForgivingIterationAdaptiveDT::computeFailedDT()
{
  _cutback_occurred = true;

  // Can't cut back any more
  if (_dt <= _dt_min)
  {
    if (_verbose)
    {
      _console << "\nSolve failed at dtmin: " << std::setw(9) << _dt_min
               << "\nReturning dtmin and marking solve as failed (will not throw error)."
               << std::endl;
    }
    else
      _console << "\nSolve failed at dtmin, returning dtmin." << std::endl;

    return _dt_min;
  }

  // Normal cutback behavior when above dtmin
  if (_verbose)
  {
    _console << "\nSolve failed with dt: " << std::setw(9) << _dt
             << "\nRetrying with reduced dt: " << std::setw(9) << _dt * _cutback_factor_at_failure
             << std::endl;
  }
  else
    _console << "\nSolve failed, cutting timestep." << std::endl;

  return _dt * _cutback_factor_at_failure;
}