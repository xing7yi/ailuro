// ForgivingIterationAdaptiveDT.h
#pragma once
#include "IterationAdaptiveDT.h"

class ForgivingIterationAdaptiveDT : public IterationAdaptiveDT
{
public:
  static InputParameters validParams();
  ForgivingIterationAdaptiveDT(const InputParameters & parameters);

protected:
  virtual Real computeFailedDT() override;
};