#include "AiluroApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "ModulesApp.h"
#include "MooseSyntax.h"

InputParameters
AiluroApp::validParams()
{
  InputParameters params = MooseApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

AiluroApp::AiluroApp(const InputParameters & parameters) : MooseApp(parameters)
{
  AiluroApp::registerAll(_factory, _action_factory, _syntax);
}

AiluroApp::~AiluroApp() {}

void
AiluroApp::registerAll(Factory & f, ActionFactory & af, Syntax & syntax)
{
  ModulesApp::registerAllObjects<AiluroApp>(f, af, syntax);
  Registry::registerObjectsTo(f, {"AiluroApp"});
  Registry::registerActionsTo(af, {"AiluroApp"});

  /* register custom execute flags, action syntax, etc. here */
}

void
AiluroApp::registerApps()
{
  registerApp(AiluroApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
extern "C" void
AiluroApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  AiluroApp::registerAll(f, af, s);
}
extern "C" void
AiluroApp__registerApps()
{
  AiluroApp::registerApps();
}
