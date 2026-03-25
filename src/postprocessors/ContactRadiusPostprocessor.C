#include "ContactRadiusPostprocessor.h"

registerMooseObject("AiluroApp", ContactRadiusPostprocessor);

InputParameters
ContactRadiusPostprocessor::validParams()
{
  auto params = NodalVariablePostprocessor::validParams();
  params.addParam<Real>("threshold", 1.0,
      "Contact pressure threshold (Pa). "
      "Nodes with cp > threshold are treated as in contact.");
  params.addClassDescription(
      "Computes the contact radius by midpoint interpolation between "
      "the outermost in-contact node and the nearest out-of-contact node, "
      "reducing the staircase artifact of the area-integral method.");
  return params;
}

ContactRadiusPostprocessor::ContactRadiusPostprocessor(const InputParameters & params)
  : NodalVariablePostprocessor(params),
    _threshold(getParam<Real>("threshold")),
    _r_in_max(-1.0),
    _r_out_min(std::numeric_limits<Real>::max()),
    _contact_radius(0.0)
{
}

void
ContactRadiusPostprocessor::initialize()
{
  _r_in_max = -1.0;
  _r_out_min = std::numeric_limits<Real>::max();
  _contact_radius = 0.0;
}

void
ContactRadiusPostprocessor::execute()
{
  // _current_node is the boundary node; _u[_qp] is the contact_pressure value
  const Real r  = (*_current_node)(0); // x = r in RZ axisymmetric
  const Real cp = _u[_qp];

  if (cp > _threshold)
    _r_in_max = std::max(_r_in_max, r);
  else
    _r_out_min = std::min(_r_out_min, r);
}

void
ContactRadiusPostprocessor::threadJoin(const UserObject & uo)
{
  const auto & pps = static_cast<const ContactRadiusPostprocessor &>(uo);
  _r_in_max  = std::max(_r_in_max,  pps._r_in_max);
  _r_out_min = std::min(_r_out_min, pps._r_out_min);
}

void
ContactRadiusPostprocessor::finalize()
{
  _communicator.max(_r_in_max);
  _communicator.min(_r_out_min);

  if (_r_in_max < 0.0)
  {
    // No contact
    _contact_radius = 0.0;
  }
  else if (_r_out_min > _r_in_max &&
           _r_out_min < std::numeric_limits<Real>::max())
  {
    // Midpoint between last in-contact and first out-of-contact node
    _contact_radius = (_r_in_max + _r_out_min) * 0.5;
  }
  else
  {
    _contact_radius = _r_in_max;
  }
}

PostprocessorValue
ContactRadiusPostprocessor::getValue() const
{
  return _contact_radius;
}
