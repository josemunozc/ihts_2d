//#include <deal.II/base/parameter_handler.h>
namespace Parameters
{
using namespace dealii;

template <int dim>
struct AllParameters
{
	AllParameters ();

	static void declare_parameters (ParameterHandler &prm);
	void parse_parameters (ParameterHandler &prm);

	double time_step;
	double theta;

	double canopy_density;
	double shading_factor;
	double thermal_conductivity_factor;
	//
	//    //    std::string surface_type;
	//    // std::string activation_type;
	std::string author;
	std::string weather_type;
	std::string input_path;

	unsigned int preheating_step;

	bool with_shading;
	bool fixed_bc_at_bottom;
	bool with_insulation;
	bool with_pipe_system;
};

template <int dim>
AllParameters<dim>::AllParameters ()
{
	time_step=0.;
	theta=0.;

	canopy_density=0.;
	shading_factor=0.;
	thermal_conductivity_factor=0.;

	preheating_step=1;

	with_shading=false;
	fixed_bc_at_bottom=false;
	with_insulation=false;
	with_pipe_system=false;
}

template <int dim>
void
AllParameters<dim>::declare_parameters (ParameterHandler &prm)
{
		  prm.enter_subsection("time stepping");
		  {
			  prm.declare_entry("time step", "3600",
					  Patterns::Double(0),
					  "simulation time step");

			  prm.declare_entry("theta", "0.5",
					  Patterns::Double(0,1),
					  "value for theta that interpolated between explicit "
					  "Euler (theta=0), Crank-Nicolson (theta=0.5), and "
					  "implicit Euler (theta=1).");
		  }
	     prm.leave_subsection();

	     prm.enter_subsection("surface conditions");
	     {
	    	 prm.declare_entry("canopy density", "0.85",
	    			 Patterns::Double(0),
					 "fraction of the soil surface covered"
					 "by vegetation.");
	     }
	     prm.leave_subsection();

	     prm.enter_subsection("options");
	     {
	    	 prm.declare_entry("author", "Best",
	    			 Patterns::Anything(),
					 "The author defines the theoretical framework used "
					 "at the soil boundary condition for heat transfer.");

	    	 prm.declare_entry("preheating step", "1",
	    			 Patterns::Integer(1,8),
	    			 "Defines the simulation step from preanalysis (1)"
	    			 "installation (2), "
	    			 "first insulation period (3),"
	    			 "fist activation period - collection(4), "
	    			 "second activation period - usage(5), "
	    			 "second insulation period (6), "
	    			 "third activation period - collection (7), "
	    			 "fourth activation period - usage (8).");

	    	 prm.declare_entry("with shading", "false",
	    			 Patterns::Bool(),
					 "Defines the use of a shading factor on the "
					 "road surface. The shading factor is defined by "
					 "shading_factor.");

	    	 prm.declare_entry("fixed bc at bottom", false,
	    			 Patterns::Bool(),
					 "Defines the state of the boundary condition at the "
					 "domain's bottom, currently two possibilities are "
					 "implemented: free, and fixed (fixed values is defined) "
					 "by fixed_bottom_bc.");

	    	 prm.declare_entry("with insulation", false,
	    			 Patterns::Bool(),
					 "Defines if the insulation layer above the storage "
					 "region should be used, if not, regular soil thermal "
					 "properties are used (i.e. the soil that composes most "
					 "the domain.");

	    	 prm.declare_entry("with pipe system", false,
	    			 Patterns::Bool(),
					 "Defines if the pipe system should be active during "
					 "the current activation period.");

	    	 prm.declare_entry("weather type", "emd-trl",
	    			 Patterns::Anything(),
					 "Defines the type of weather to be used, this is, based "
					 "on analytical expressions or on meteorological measurements; "
					 "and if based on in-situ measurements or met-office data for "
					 "the period 2005-2007."
					 "Options: amd-mild-1 (analytical based on UK TRL in-situ data), "
					 "amd-mild-2 (analytical based on UK met office data), "
					 "amd-cold (analytical based on iceland met office data), "
					 "amd-hot (analytical based on Yucatan,Mexico met office data (FIUADY)),"
					 "emd-trl (experimental based on UK TRL in-situ data), "
					 "emd-badc (experimental based on UK BADC data for Toddington, UK)");

	    	 prm.declare_entry("shading factor", "0.0",
	    			 Patterns::Double(0.0,1.0),
					 "Defines the amount of shading on the road surface. "
					 "The solar radiation term is multiplied by (1-shading_factor), "
					 "this means, for example: shading_factor= 1 (total shading, "
					 "no solar radiation), shading_factor= 0 (no shading, total "
					 "solar radiation).");

	    	 prm.declare_entry("thermal conductivity factor", "1.0",
	    			 Patterns::Double(0.1,2.0), "Defines the weighting factor for the "
	    					 "thermal conductivity in the storage region. This "
	    					 "is the region defined as (in 2D): -6m<x<6m, -0.75m>y>-9m. "
	    					 "The thermal conductivity is multiplied by this factor so "
	    					 "that: 1.0 (no change in base thermal conductivity), "
	    					 "0.9 (90% of base thermal conductivity). This factor only "
	    					 "affects the soil material and is currently implemented for "
	    					 "activation periods >=4. and for changes between 10% -- 200%.");

	    	 prm.declare_entry("inputh path", "/home/zerpiko/input/",
	    			 Patterns::Anything(),
					 "Define the top level directory with input files such as "
					 "meteorological data or mesh files. These files are assumed to "
					 "be sorted in files under this path.");
	     }
	     prm.leave_subsection();
}

template <int dim>
void AllParameters<dim>::parse_parameters (ParameterHandler &prm)
{
		  prm.enter_subsection("time stepping");
		  {
			  time_step           = prm.get_double ("time step");
			  theta               = prm.get_double ("theta");
		  }
		  prm.leave_subsection();

		  prm.enter_subsection("surface conditions");
		  {
			  canopy_density = prm.get_double ("canopy density");
		  }

		  prm.enter_subsection("options");
		  {
			  author             = prm.get         ("author");
			  preheating_step    = prm.get_integer ("preheating step");
			  with_shading       = prm.get_bool    ("with shading");
			  fixed_bc_at_bottom = prm.get_bool ("fixed bc at bottom");
			  with_insulation    = prm.get_bool ("with insulation");
			  with_pipe_system   = prm.get_bool ("with pipe system");
			  weather_type       = prm.get      ("weather type");
			  shading_factor     = prm.get_double ("shading factor");
			  thermal_conductivity_factor = prm.get_double ("thermal conductivity factor");
			  input_path         = prm.get      ("input path");
		  }
}
}
