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
      "sigma_p = sigma_0 * (1 + eps_p / eps_0)^n, "
      "where sigma_0 is the yield strength.");

  // sigma_0 replaces the parent's yield_stress
  params.addRequiredParam<Real>("sigma_0", "Yield strength / initial flow stress (sigma_0)");
  params.addRequiredParam<Real>("eps_0", "Reference plastic strain (eps_0)");
  params.addRequiredParam<Real>("n", "Hardening exponent (n)");

  // Suppress parent params that are superseded by the Swift parameters above
  params.set<Real>("yield_stress") = 1.0;
  params.suppressParameter<Real>("yield_stress");
  params.set<Real>("hardening_constant") = 0.0;
  params.suppressParameter<Real>("hardening_constant");

  params.declareControllable("sigma_0 eps_0 n");
  return params;
}

template <bool is_ad>
IsotropicSwiftLawHardeningStressUpdateTempl<is_ad>::IsotropicSwiftLawHardeningStressUpdateTempl(
    const InputParameters & parameters)
  : IsotropicLinearHardeningStressUpdateTempl<is_ad>(parameters),
    _sigma_0(this->template getParam<Real>("sigma_0")),
    _eps_0(this->template getParam<Real>("eps_0")),
    _n(this->template getParam<Real>("n"))
{
}

// ---------------------------------------------------------------------------
// Set yield stress = sigma_0 (called once per quadrature point from
// computeStressInitialize in the parent class)
// ---------------------------------------------------------------------------

template <bool is_ad>
void
IsotropicSwiftLawHardeningStressUpdateTempl<is_ad>::computeYieldStress(
    const GenericRankFourTensor<is_ad> & /*elasticity_tensor*/)
{
  _yield_stress = _sigma_0;
}

// ---------------------------------------------------------------------------
// Hardening variable H = sigma_0 * [(1 + eps_p/eps_0)^n - 1]
//
// Total flow stress = yield_stress + H
//                   = sigma_0 + sigma_0*[(1+eps_p/eps_0)^n - 1]
//                   = sigma_0 * (1 + eps_p/eps_0)^n     (= sigma_p)
// ---------------------------------------------------------------------------

template <bool is_ad>
GenericReal<is_ad>
IsotropicSwiftLawHardeningStressUpdateTempl<is_ad>::computeHardeningValue(
    const GenericReal<is_ad> & scalar)
{
  using std::pow;
  const GenericReal<is_ad> total_eps = _effective_inelastic_strain_old[_qp] + scalar;
  // Store for yield-condition bookkeeping used by parent
  _hardening_variable[_qp] = _sigma_0 * (pow(1.0 + total_eps / _eps_0, _n) - 1.0);
  return _hardening_variable[_qp];
}

// ---------------------------------------------------------------------------
// dH/d(eps_p) = sigma_0 * n / eps_0 * (1 + eps_p/eps_0)^(n-1)
// ---------------------------------------------------------------------------

template <bool is_ad>
GenericReal<is_ad>
IsotropicSwiftLawHardeningStressUpdateTempl<is_ad>::computeHardeningDerivative(
    const GenericReal<is_ad> & scalar)
{
  using std::pow;
  const GenericReal<is_ad> total_eps = _effective_inelastic_strain_old[_qp] + scalar;
  return _sigma_0 * _n / _eps_0 * pow(1.0 + total_eps / _eps_0, _n - 1.0);
}

template class IsotropicSwiftLawHardeningStressUpdateTempl<false>;
template class IsotropicSwiftLawHardeningStressUpdateTempl<true>;