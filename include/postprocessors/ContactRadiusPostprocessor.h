#pragma once

#include "NodalVariablePostprocessor.h"

/**
 * Computes contact radius with sub-element smoothing.
 *
 * Usage: set boundary = <secondary> in the input block to restrict evaluation
 * to the contact boundary.  The base class iterates over boundary nodes
 * automatically, calling execute() once per node.
 *
 * Tracks:
 *   r_in  = max radial coord of in-contact nodes  (cp > threshold)
 *   r_out = min radial coord of out-of-contact nodes (cp <= threshold, r > 0)
 *
 * Returns (r_in + r_out) / 2 when both exist, otherwise r_in.
 * This halves the staircase amplitude versus the area-based estimate.
 */
class ContactRadiusPostprocessor : public NodalVariablePostprocessor
{
public:
  static InputParameters validParams();
  ContactRadiusPostprocessor(const InputParameters & params);

  void initialize() override;
  void execute() override;
  void threadJoin(const UserObject & uo) override;
  void finalize() override;
  PostprocessorValue getValue() const override;

protected:
  const Real _threshold;

  Real _r_in_max;
  Real _r_out_min;
  Real _contact_radius;
};
