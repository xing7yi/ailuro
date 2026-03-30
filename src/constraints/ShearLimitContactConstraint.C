//* This file is part of the ailuro application

#include "ShearLimitContactConstraint.h"
#include "PenetrationInfo.h"

registerMooseObject("AiluroApp", ShearLimitContactConstraint);

InputParameters
ShearLimitContactConstraint::validParams()
{
  InputParameters params = MechanicalContactConstraint::validParams();
  params.addClassDescription(
      "Coulomb friction with shear stress limit: "
      "tau = min(mu * p, shear_limit). "
      "Use with formulation = TANGENTIAL_PENALTY, model = COULOMB.");
  params.addParam<Real>("shear_limit", -1.0,
      "Maximum allowed shear traction (stress units, e.g. cohesion c). "
      "Negative value (default): disabled, degrades to standard Coulomb friction.");
  params.addParam<Real>("yield_stress_for_shear", -1.0,
      "When > 0, overrides shear_limit with yield_stress_for_shear / sqrt(3) (von Mises criterion). "
      "Declare the same sampler column as Materials yield_stress to keep them in sync.");
  params.declareControllable("shear_limit yield_stress_for_shear");
  return params;
}

ShearLimitContactConstraint::ShearLimitContactConstraint(const InputParameters & parameters)
  : MechanicalContactConstraint(parameters),
    _shear_limit(getParam<Real>("shear_limit")),
    _ys_for_shear(getParam<Real>("yield_stress_for_shear"))
{
  if (_formulation != ContactFormulation::TANGENTIAL_PENALTY ||
      _model != ContactModel::COULOMB)
    mooseWarning("ShearLimitContactConstraint: shear_limit is only applied to "
                 "TANGENTIAL_PENALTY + COULOMB.");
}

bool
ShearLimitContactConstraint::shouldApply()
{
  bool in_contact = false;

  auto found = _penetration_locator._penetration_info.find(_current_node->id());
  if (found != _penetration_locator._penetration_info.end())
  {
    PenetrationInfo * pinfo = found->second;
    if (pinfo != NULL)
    {
      bool is_nonlinear = _subproblem.computingNonlinearResid();

      if (_component == 0)
      {
        // Standard contact force computation
        computeContactForce(*_current_node, pinfo, is_nonlinear);
        // Apply shear limit post-processing
        applyShearLimit(*_current_node, pinfo);
      }

      if (pinfo->isCaptured())
      {
        in_contact = true;
        if (is_nonlinear)
        {
          Threads::spin_mutex::scoped_lock lock(_contact_set_mutex);
          _current_contact_state.insert(_current_node->id());
        }
      }
    }
  }

  return in_contact;
}

void
ShearLimitContactConstraint::applyShearLimit(const Node & node, PenetrationInfo * pinfo)
{
  // _ys_for_shear > 0: use von Mises formula; else use explicit _shear_limit.
  // Either way, a non-positive effective limit means disabled (pure Coulomb).
  const Real effective_limit = (_ys_for_shear > 0.0)
                                   ? _ys_for_shear / std::sqrt(3.0)
                                   : _shear_limit;
  if (effective_limit <= 0.0 || !pinfo->isCaptured())
    return;

  // Decompose force into normal and tangential
  const RealVectorValue fn = (pinfo->_contact_force * pinfo->_normal) * pinfo->_normal;
  const RealVectorValue ft = pinfo->_contact_force - fn;
  const Real tan_mag = ft.norm();

  // Shear force limit = effective shear traction * nodal_area
  const Real shear_force_cap = effective_limit * nodalArea(node);

  if (tan_mag > shear_force_cap && tan_mag > 0.0)
  {
    pinfo->_contact_force = fn + (shear_force_cap / tan_mag) * ft;
    pinfo->_mech_status = PenetrationInfo::MS_SLIPPING_FRICTION;
  }
}

