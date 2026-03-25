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
  return params;
}

ShearLimitContactConstraint::ShearLimitContactConstraint(const InputParameters & parameters)
  : MechanicalContactConstraint(parameters), _shear_limit(getParam<Real>("shear_limit"))
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
  // _shear_limit <= 0 means disabled: pure Coulomb, no shear cap
  if (_shear_limit <= 0.0 || !pinfo->isCaptured())
    return;

  // Decompose force into normal and tangential
  const RealVectorValue fn = (pinfo->_contact_force * pinfo->_normal) * pinfo->_normal;
  const RealVectorValue ft = pinfo->_contact_force - fn;
  const Real tan_mag = ft.norm();

  // Shear force limit = shear_limit_traction * nodal_area
  const Real shear_force_cap = _shear_limit * nodalArea(node);

  if (tan_mag > shear_force_cap && tan_mag > 0.0)
  {
    pinfo->_contact_force = fn + (shear_force_cap / tan_mag) * ft;
    pinfo->_mech_status = PenetrationInfo::MS_SLIPPING_FRICTION;
  }
}

