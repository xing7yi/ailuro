#pragma once

#include "CombinedNonlinearHardeningPlasticity.h"

template <bool is_ad>
class TestControllableWithoutComputeYieldStressTempl
  : public CombinedNonlinearHardeningPlasticityTempl<is_ad>
{
public:
  static InputParameters validParams();

  TestControllableWithoutComputeYieldStressTempl(const InputParameters & parameters);

  using Material::_qp;
  using RadialReturnBackstressStressUpdateBaseTempl<is_ad>::_base_name;

protected:
  // 注意：我们故意不重写computeYieldStress函数
  
  // Controllable parameters as const references
  const Real & _controllable_yield_stress;
  const Real & _controllable_isotropic_hardening_constant;
};

typedef TestControllableWithoutComputeYieldStressTempl<false> TestControllableWithoutComputeYieldStress;
typedef TestControllableWithoutComputeYieldStressTempl<true> ADTestControllableWithoutComputeYieldStress;
