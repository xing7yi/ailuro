#include "IsotropicSwiftLawHardeningStressUpdate.h"

registerMooseObject("AiluroApp", ADIsotropicSwiftLawHardeningStressUpdate);
registerMooseObject("AiluroApp", IsotropicSwiftLawHardeningStressUpdate);

template <bool is_ad>
InputParameters
IsotropicSwiftLawHardeningStressUpdateTempl<is_ad>::validParams()
{
  InputParameters params = IsotropicLinearHardeningStressUpdateTempl<is_ad>::validParams();
  params.addClassDescription(
      "Isotropic plasticity with Swift power law hardening: "
      "sigma_p = yield_stress * (1 + eps_p / eps_0)^n.");

  params.addRequiredParam<Real>("eps_0", "Reference plastic strain (eps_0)");
  params.addRequiredParam<Real>("n", "Hardening exponent (n)");

  // hardening_constant is not used; suppress it and set a dummy default
  params.set<Real>("hardening_constant") = 0.0;
  params.suppressParameter<Real>("hardening_constant");
  params.declareControllable("eps_0 n");
  return params;
}

template <bool is_ad>
IsotropicSwiftLawHardeningStressUpdateTempl<is_ad>::IsotropicSwiftLawHardeningStressUpdateTempl(
    const InputParameters & parameters)
  : IsotropicLinearHardeningStressUpdateTempl<is_ad>(parameters),
    _eps_0(this->template getParam<Real>("eps_0")),
    _n(this->template getParam<Real>("n"))
{
}

// ---------------------------------------------------------------------------
// Hardening variable H = yield_stress * [(1 + eps_p/eps_0)^n - 1]
//
// Total flow stress = yield_stress + H
//                   = yield_stress * (1 + eps_p/eps_0)^n     (= sigma_p)
// ---------------------------------------------------------------------------

template <bool is_ad>
GenericReal<is_ad>
IsotropicSwiftLawHardeningStressUpdateTempl<is_ad>::computeHardeningValue(
    const GenericReal<is_ad> & scalar)
{
  using std::pow;
  const GenericReal<is_ad> total_eps = _effective_inelastic_strain_old[_qp] + scalar;
  _hardening_variable[_qp] = _yield_stress * (pow(1.0 + total_eps / _eps_0, _n) - 1.0);
  return _hardening_variable[_qp];
}

// ---------------------------------------------------------------------------
// dH/d(eps_p) = yield_stress * n / eps_0 * (1 + eps_p/eps_0)^(n-1)
// ---------------------------------------------------------------------------

template <bool is_ad>
GenericReal<is_ad>
IsotropicSwiftLawHardeningStressUpdateTempl<is_ad>::computeHardeningDerivative(
    const GenericReal<is_ad> & scalar)
{
  using std::pow;
  const GenericReal<is_ad> total_eps = _effective_inelastic_strain_old[_qp] + scalar;
  return _yield_stress * _n / _eps_0 * pow(1.0 + total_eps / _eps_0, _n - 1.0);
}

template class IsotropicSwiftLawHardeningStressUpdateTempl<false>;
template class IsotropicSwiftLawHardeningStressUpdateTempl<true>;