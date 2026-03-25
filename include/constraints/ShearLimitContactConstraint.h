//* This file is part of the ailuro application

#pragma once

#include "MechanicalContactConstraint.h"
#include "ContactAction.h"

/**
 * Coulomb friction contact with a shear stress limit.
 * Inherits standard TANGENTIAL_PENALTY + COULOMB behavior
 * and caps: tau <= min(mu * p, shear_limit)
 */
class ShearLimitContactConstraint : public MechanicalContactConstraint
{
public:
  static InputParameters validParams();
  ShearLimitContactConstraint(const InputParameters & parameters);

  bool shouldApply() override;

protected:
  void applyShearLimit(const Node & node, PenetrationInfo * pinfo);

  /// Maximum shear traction (stress units)
  const Real _shear_limit;
};
