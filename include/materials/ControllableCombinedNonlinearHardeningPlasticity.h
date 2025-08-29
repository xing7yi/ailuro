#pragma once

#include "CombinedNonlinearHardeningPlasticity.h"

template <bool is_ad>
class ControllableCombinedNonlinearHardeningPlasticityTempl
  : public CombinedNonlinearHardeningPlasticityTempl<is_ad>
{
public:
  static InputParameters validParams();

  ControllableCombinedNonlinearHardeningPlasticityTempl(const InputParameters & parameters);

  using Material::_qp;
  using RadialReturnBackstressStressUpdateBaseTempl<is_ad>::_base_name;
  using RadialReturnBackstressStressUpdateBaseTempl<is_ad>::_three_shear_modulus;

protected:
  virtual void computeYieldStress(const GenericRankFourTensor<is_ad> & elasticity_tensor) override;

  // Controllable parameters as const references
  const Real & _controllable_yield_stress;
  const Real & _controllable_isotropic_hardening_constant;
};

typedef ControllableCombinedNonlinearHardeningPlasticityTempl<false> ControllableCombinedNonlinearHardeningPlasticity;
typedef ControllableCombinedNonlinearHardeningPlasticityTempl<true> ADControllableCombinedNonlinearHardeningPlasticity;
