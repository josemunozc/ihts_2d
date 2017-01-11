#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/smartpointer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
//#include <deal.II/lac/compressed_sparsity_pattern.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_bicgstab.h> 
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
//#include <deal.II/lac/compressed_simple_sparsity_pattern.h>

//#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
//#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h> 
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_accessor.templates.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <sstream> 
#include <math.h>
#include <map>
#include <sys/stat.h>

#include "Names.h"
#include "SurfaceCoefficients.h"
#include "AnalyticSolution.h"
#include "BoundaryConditions.h"
#include "DataTools.h"
#include "MaterialData.h"

namespace TRL
{
  using namespace dealii;
#include "PipeConvectiveCoefficient.h"
#include "PipeSystem.h"
  
  template <int dim>
  class Heat_Pipe
  {
  public:
    Heat_Pipe(int argc, char *argv[], FE_Q<dim> &fe_);
    ~Heat_Pipe();
    void run();
  private:
    void output_results();
     void fill_output_vectors();
    void update_met_data();
    // /*--------------------temperature functions-------------------------*/
    void read_mesh_temperature();
    void initial_condition_temperature();
    void setup_system_temperature();
    void assemble_system_petsc_temperature();
    void assemble_system_parallel_temperature(bool system_switch);
    unsigned int solve_temperature();
    // /*--------------------temperature on surfaces----------------------*/
    void surface_temperatures();
    void mesh_info();
    /*-----------------------temperature variables--------------------*/
    Triangulation<dim>   triangulation_temperature;
    DoFHandler<dim>      dof_handler_temperature;
    SmartPointer<const FiniteElement<dim> > fe_temperature;
    ConstraintMatrix     constraints_temperature;
    PETScWrappers::MPI::SparseMatrix system_matrix_temperature;
    PETScWrappers::MPI::SparseMatrix mass_matrix_temperature;
    PETScWrappers::MPI::SparseMatrix laplace_matrix_new_temperature;
    PETScWrappers::MPI::SparseMatrix laplace_matrix_old_temperature;
    PETScWrappers::MPI::Vector       system_rhs_temperature;
    PETScWrappers::MPI::Vector solution_temperature, 
      old_solution_temperature, backup_solution_temperature; 
    std::vector<types::global_dof_index> local_dofs_per_process_temperature;
    types::global_dof_index n_local_dofs_temperature;
    /*----------------1D pipe system in 3D variables---------------*/    
    std::map<typename DoFHandler<1>::active_cell_iterator,
	     std::vector< typename DoFHandler<3>::active_cell_iterator > >
    cell_index_1d_to_cell_index_3d;
    std::map<typename DoFHandler<3>::active_cell_iterator,double >
    cell_index_3d_to_old_pipe_heat_flux,
      cell_index_3d_to_previous_new_pipe_heat_flux,
      cell_index_3d_to_current__new_pipe_heat_flux;

    double inlet_collector_temperature;
    double inlet_storage_temperature;
    /*-----------------global variables-----------------*/
    ConditionalOStream pcout;
    unsigned int n_local_cells;
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    
    unsigned int timestep_number,timestep_number_max;
    double time, time_step, time_max;
    const double theta_temperature;

    const std::string surface_type;
    std::string met_data_type;
    const std::string activation_type;
    double canopy_density;
    const bool refine_mesh;
    const bool moisture_movement;

    std::string author;
    std::string preheating_output_filename;
    std::string preheating_input_filename;
    std::string input_path;
    bool fixed_bc_at_bottom;
    bool pipe_system;
    bool preheating;
    bool shading;
    bool insulation;
    bool analytic_met_data;
    int  preheating_step;
    double shading_factor_value;
    double thermal_conductivity_factor;
    int type_of_weather;
    
    const unsigned int number_of_pipes;//=40;
    const unsigned int n_boundary_ids;//=8;
    const unsigned int boundary_id_collector;//=1;
    const unsigned int boundary_id_storage;//=2;
    const unsigned int boundary_id_collector_lids;//=1;
    const unsigned int boundary_id_storage_lids;//=2;
    const unsigned int boundary_id_road;//=3;
    const unsigned int boundary_id_soil;//=4;
    const unsigned int boundary_id_soil_bottom;//=5;
    /*----------------------mesh data vectors--------------------*/
    std::map<typename DoFHandler<dim>::active_cell_iterator,unsigned int >
    cell_index_to_face_index,    // relate cells at boundary with corresponding face at boundary
      cell_index_to_mpi_process, // to distribute calculations on boundary cells on all mpi processes
      cell_index_to_pipe_number; // relate boundary cells on pipes with its pipe number (0-19)
    std::map<typename DoFHandler<dim>::active_cell_iterator,Point<dim> >
    cell_index_to_cell_center;
    std::map<typename DoFHandler<dim>::active_cell_iterator,double >
    cell_index_to_old_surface_temperature,
      cell_index_to_current__new_surface_temperature,
      cell_index_to_previous_new_surface_temperature;
    std::map<double,std::map<double,typename DoFHandler<dim>::active_cell_iterator > >
    y_coordinates_to_x_coordinates_to_cell_indexes;    
    /*-------------met data vectors and variables----------------*/
    std::vector< std::vector<int> >    date_and_time;
    std::vector< std::vector<double> > met_data;
    std::vector<int> initial_date;
    double new_air_temperature,old_air_temperature;
    double new_relative_humidity,old_relative_humidity;
    double new_wind_speed,old_wind_speed;
    double new_wind_direction,old_wind_direction;
    double new_solar_radiation,old_solar_radiation;
    double new_precipitation,old_precipitation;
    /*--------------pipe system vectors and variables--------------*/
    std::vector<double> max_pipe_temperature;
    std::vector<double> new_avg_pipe_temperature;
    std::vector<double> old_avg_pipe_temperature;
    std::vector<double> min_pipe_temperature;
    std::vector<int> cell_faces_per_pipe;

    std::vector<double> old_pipe_heat_flux;
    std::vector<double> current__new_pipe_heat_flux;

    double old_avg_soil_surface_temperature;
    double old_avg_road_surface_temperature;
    double old_max_soil_surface_temperature;
    double old_max_road_surface_temperature;
    double old_min_soil_surface_temperature;
    double old_min_road_surface_temperature;

    double previous_new_soil_avg_surface_temperature;
    double previous_new_road_avg_surface_temperature;
    double previous_new_soil_max_surface_temperature;
    double previous_new_soil_min_surface_temperature;
    double previous_new_road_max_surface_temperature;
    double previous_new_road_min_surface_temperature;

    double current_new_avg_soil_surface_temperature;
    double current_new_avg_road_surface_temperature;
    double current_new_max_soil_surface_temperature;
    double current_new_max_road_surface_temperature;
    double current_new_min_soil_surface_temperature;
    double current_new_min_road_surface_temperature;
    
    const double collector_depth;//=0.1325;
    const double storage_depth;//  =0.8475;
    /*------------ vectors for output data --------------*/
    const unsigned int number_of_surface_heat_and_mass_fluxes;//=11;
    //static const Point<dim> stores_centers[2];
    Point<dim> borehole_A_depths[35];
    Point<dim> borehole_F_depths[35];
    Point<dim> borehole_H_depths[35];
    Point<dim> borehole_I_depths[35];
    
    std::vector< std::vector<int> > date_and_time_1d;
    
    std::vector< std::vector<double> > soil_bha_temperature;
    std::vector< std::vector<double> > soil_bhf_temperature;
    std::vector< std::vector<double> > soil_bhh_temperature;
    std::vector< std::vector<double> > soil_bhi_temperature;
    std::vector< std::vector<double> > road_heat_fluxes;
    std::vector< std::vector<double> > soil_heat_fluxes;
    std::vector< std::vector<double> > pipe_heat_fluxes;
    std::vector< std::vector<double> > control_temperatures;

    static std::string author_options[3];
  };

  template<int dim>
  std::string Heat_Pipe<dim>::author_options[3]={"Herb","Best","Fixed"};
  
  template<int dim>
  Heat_Pipe<dim>::Heat_Pipe (int argc, char *argv[], FE_Q<dim> &fe_)
    :
    dof_handler_temperature (triangulation_temperature),
    fe_temperature          (&fe_),
    pcout               (std::cout),
    mpi_communicator    (MPI_COMM_WORLD),
    n_mpi_processes     (Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process    (Utilities::MPI::this_mpi_process(mpi_communicator)),
    /*
      Preheating     timestep     time_max     details
      1st             3600         35027        normal - (3 years)
      2nd             3600          5804        normal
      3rd             3600          2735        with insulation
      3rd             1800          5470        with insulation
      4th              900          8060        system on
      5th              900          8060        system on
    */
    time_step        (3600),
    theta_temperature(0.5),
    surface_type     ("soil"),
    activation_type  ("automatic_activation"),
    canopy_density   (0.85),
    refine_mesh      (false),
    moisture_movement(false),
    number_of_pipes  (40),
    n_boundary_ids            (8),
    boundary_id_collector     (1),
    boundary_id_storage       (2),
    boundary_id_collector_lids(1),
    boundary_id_storage_lids  (2),
    boundary_id_road          (3),
    boundary_id_soil          (4),
    boundary_id_soil_bottom   (5),
    max_pipe_temperature       (number_of_pipes,-1.e6),
    new_avg_pipe_temperature   (number_of_pipes,0.),
    old_avg_pipe_temperature   (number_of_pipes,0.),
    min_pipe_temperature       (number_of_pipes,1.e6),
    cell_faces_per_pipe        (number_of_pipes,0),
    old_pipe_heat_flux         (number_of_pipes,0),
    current__new_pipe_heat_flux(number_of_pipes,0),
    collector_depth            (0.1325),
    storage_depth              (0.8475),
    number_of_surface_heat_and_mass_fluxes(11)
  {
    pcout.set_condition(this_mpi_process==0);
    std::string shading_factor_value_str;
    std::string thermal_conductivity_factor_str;
    int number_of_arguments=11;
    
    if (argc==number_of_arguments)
      {
	/*
	 * amd: analytical meteorological data
	 * emd: experimental meteorological data
	 */
	std::string condition_false="false";
	std::string condition_true="true";
	std::string type_of_weather_1 = "amd-mild-1";
	std::string type_of_weather_2 = "amd-mild-2";
	std::string type_of_weather_3 = "amd-cold";
	std::string type_of_weather_4 = "amd-hot";
	std::string type_of_weather_5 = "emd-trl";
	std::string type_of_weather_6 = "emd-badc";
	author=argv[1];
	preheating_step=atoi(argv[2]);
	std::string shading_str=argv[3];
	std::string bottom_bc=argv[4];
	std::string type_of_weather_str=argv[5];
	std::string insulation_str=argv[6];
	std::string pipe_system_str=argv[7];
	shading_factor_value_str=argv[8];
	thermal_conductivity_factor_str=argv[9];
	input_path=argv[10];
	
	if ((strcmp(author.c_str(),author_options[0].c_str())!=0) &&
	    (strcmp(author.c_str(),author_options[1].c_str())!=0) &&
	    (strcmp(author.c_str(),author_options[2].c_str())!=0))
	  {
	    pcout << "wrong author. current input: " << author << "\n";
	  }
	
	if (strcmp(shading_str.c_str(),condition_true.c_str())==0)
	  shading=true;
	else if (strcmp(shading_str.c_str(),condition_false.c_str())==0)
	  shading=false;
	else
	  {
	    pcout << "wrong shading condition\n";
	    throw -1;
	  }

	if (strcmp(bottom_bc.c_str(),condition_true.c_str())==0)
	  fixed_bc_at_bottom=true;
	else if (strcmp(bottom_bc.c_str(),condition_false.c_str())==0)
	  fixed_bc_at_bottom=false;
	else
	  {
	    pcout << "wrong bottom_bc\n";
	    throw -1;
	  }

	if (strcmp(insulation_str.c_str(),condition_true.c_str())==0)
	  insulation=true;
	else if (strcmp(insulation_str.c_str(),condition_false.c_str())==0)
	  insulation=false;
	else
	  {
	    pcout << "wrong insulation condition\n";
	    throw -1;
	  }

	if (strcmp(pipe_system_str.c_str(),condition_true.c_str())==0)
	  pipe_system=true;
	else if (strcmp(pipe_system_str.c_str(),condition_false.c_str())==0)
	  pipe_system=false;
	else
	  {
	    pcout << "wrong pipe system condition\n";
	    throw -1;
	  }
	
	if (strcmp(type_of_weather_str.c_str(),type_of_weather_1.c_str())==0)
	  {
	    analytic_met_data=true;
	    type_of_weather=1;
	  }
	else if (strcmp(type_of_weather_str.c_str(),type_of_weather_2.c_str())==0)
	  {
	    analytic_met_data=true;
	    type_of_weather=2;
	  }
	else if (strcmp(type_of_weather_str.c_str(),type_of_weather_3.c_str())==0)
	  {
	    analytic_met_data=true;
	    type_of_weather=3;
	  }
	else if (strcmp(type_of_weather_str.c_str(),type_of_weather_4.c_str())==0)
	  {
	    analytic_met_data=true;
	    type_of_weather=4;
	  }
	else if (strcmp(type_of_weather_str.c_str(),type_of_weather_5.c_str())==0)
	  {
	    analytic_met_data=false;
	    type_of_weather=-1;
	    met_data_type="trl_met_data";
	  }
	else if (strcmp(type_of_weather_str.c_str(),type_of_weather_6.c_str())==0)
	  {
	    analytic_met_data=false;
	    type_of_weather=-1;
	    met_data_type="met_office_data";
	  }
	else
	  {
	    pcout << "Error. Wrong type of weather.\n";
	    throw -1;
	  }
	
	shading_factor_value=atof(shading_factor_value_str.c_str());
	if (shading_factor_value>1. ||
	    shading_factor_value<0.)
	  {
	    pcout << "wrong shading factor value\n";
	    throw -1;
	  }
	
	thermal_conductivity_factor=atof(thermal_conductivity_factor_str.c_str());
	if (thermal_conductivity_factor<0. ||
	    thermal_conductivity_factor>2.)
	  {
	    pcout << "wrong thermal conductivity factor value\n";
	    throw -1;
	  }
      }
    else
      {
	pcout << "Not enough parameters\n"
	      << "prog_name\t"
	      << "author_name\t"
	      << "preheating_step\t"
	      << "shading\t"
	      << "bottom_bc\t"
	      << "type_of_weather\t"
	      << "insulation\t"
	      << "pipe_system\t"
	      << "shading_factor_value\t"
	      << "thermal_conductivity_factor\t\n";
	pcout << "Arguments passed: " << argc << "\n";
	for (int i=0; i<argc; i++)
	  pcout << "\t" << argv[i];
	pcout << "\n";
	
	throw -1;
      }
     
    std::stringstream out_phs;
    out_phs << preheating_step;
    std::stringstream in_phs;
    in_phs << preheating_step-1;
    
    preheating_output_filename=author+"_ph"+out_phs.str()+"_2d_";
    preheating_input_filename=author+"_ph"+in_phs.str()+"_2d_";
    
    if (shading==true)
      {
	preheating_output_filename+="ws_";
	preheating_input_filename+="ws_";
      }
    else
      {
	preheating_output_filename+="wos_";
	preheating_input_filename+="wos_";
      }
    
    if (fixed_bc_at_bottom==true)
      {
	preheating_output_filename+="fx_";
	preheating_input_filename+="fx_";
      }
    else
      {
	preheating_output_filename+="fr_";
	preheating_input_filename+="fr_";
      }
    
    if (type_of_weather==1)
      {
	preheating_output_filename+="amd-mild-1_";
	preheating_input_filename+="amd-mild-1_";
      }
    else if (type_of_weather==2)
      {
	preheating_output_filename+="amd-mild-2_";
	preheating_input_filename+="amd-mild-2_";
      }
    else if (type_of_weather==3)
      {
	preheating_output_filename+="amd-cold_";
	preheating_input_filename+="amd-cold_";
      }
    else if (type_of_weather==4)
      {
	preheating_output_filename+="amd-hot_";
	preheating_input_filename+="amd-hot_";
      }
    else if (type_of_weather==-1)
      {
	  preheating_output_filename+="emd-trl_";
	  preheating_input_filename+="emd-trl_";
      }
    
    preheating_output_filename+=shading_factor_value_str+"_";
    preheating_input_filename+=shading_factor_value_str+"_";

    preheating_output_filename+=thermal_conductivity_factor_str;
    preheating_input_filename+=thermal_conductivity_factor_str;
    
    if (preheating_step==1 && time_step==3600)
      {
	timestep_number_max=70079; // 8 years
	initial_date.reserve(6);
	initial_date.push_back(1);
	initial_date.push_back(9);
	initial_date.push_back(1994);
	initial_date.push_back(0);
	initial_date.push_back(30);
	initial_date.push_back(0);
      }
    else if (preheating_step==2 && time_step==3600)
      {
	timestep_number_max=5804;
	initial_date.reserve(6);
	initial_date.push_back(1);
	initial_date.push_back(9);
	initial_date.push_back(2004);
	initial_date.push_back(0);
	initial_date.push_back(30);
	initial_date.push_back(0);
      }
    else if (preheating_step==3 && time_step==3600)
      {
	timestep_number_max=2735;
	initial_date.reserve(6);
	initial_date.push_back(1);
	initial_date.push_back(5);
	initial_date.push_back(2005);
	initial_date.push_back(0);
	initial_date.push_back(30);
	initial_date.push_back(0);
      }
    else if (preheating_step==4)
      {
	time_step=900;
	timestep_number_max=7964;
	initial_date.reserve(6);
	initial_date.push_back(23);
	initial_date.push_back(8);
	initial_date.push_back(2005);
	initial_date.push_back(0);
	initial_date.push_back(30);
	initial_date.push_back(0);
      }
    else if (preheating_step==5)
      {
	time_step=900;
	timestep_number_max=9500;
	initial_date.reserve(6);
	initial_date.push_back(14);
	initial_date.push_back(11);
	initial_date.push_back(2005);
	initial_date.push_back(0);
	initial_date.push_back(30);
	initial_date.push_back(0);
      }
    else if (preheating_step==6)
      {
	time_step=3600;
	timestep_number_max=1559;
	initial_date.reserve(6);
	initial_date.push_back(21);
	initial_date.push_back(2);
	initial_date.push_back(2006);
	initial_date.push_back(0);
	initial_date.push_back(30);
	initial_date.push_back(0);
      }
    else if (preheating_step==7)
      {
	time_step=900;
	timestep_number_max=18044;
	initial_date.reserve(6);
	initial_date.push_back(27);
	initial_date.push_back(4);
	initial_date.push_back(2006);
	initial_date.push_back(0);
	initial_date.push_back(30);
	initial_date.push_back(0);
      }
    else if (preheating_step==8)
      {
	time_step=900;
	timestep_number_max=11516;
	initial_date.reserve(6);
	initial_date.push_back(1);
	initial_date.push_back(11);
	initial_date.push_back(2006);
	initial_date.push_back(0);
	initial_date.push_back(30);
	initial_date.push_back(0);
      }
    else
      {
	pcout << "Wrong preheating step and/or time step\n";
	throw -1;
      }
    time_max=time_step*timestep_number_max;

    if (preheating_step>1)
      preheating=true;
    else
      preheating=false;

    pcout << "Solving problem with the following data:\n"
	  << "\tAuthor: " << author << "\n"
	  << "\tpreheating_step: " << preheating_step << "\n"
	  << "\ttime step: " << time_step << "\n"
	  << "\tshading: ";
    if (shading==true)
      pcout << "true\n";
    else
      pcout << "false\n";

    pcout << "\tshading_factor_value: " << shading_factor_value << "\n";
    pcout << "\tbottom_bc: ";
    if (fixed_bc_at_bottom==true)
      pcout << "true\n";
    else
      pcout << "false\n";

    pcout << "\tinsulation: ";
    if (insulation==true)
      pcout << "true\n";
    else
      pcout << "false\n";
    
    pcout << "\tpipe_system: ";
    if (pipe_system==true)
      pcout << "true\n";
    else
      pcout << "false\n";

    pcout << "\tOutput preheating file: " << preheating_output_filename << "\n";
    if (preheating)
      pcout << "\tInput preheating file : " << preheating_input_filename << "\n";
    else
      pcout << "No input preheating file defined\n";


    DataTools data_tools;
    if (!data_tools.dir_exists(input_path))
      std::cout << "Error. Inputh path: " << input_path << "doesn't exists.\n";
    else
      pcout << "Input path: " << input_path << std::endl;
    
    // for (unsigned int i=0; i<(sizeof borehole_A_depths)/(sizeof borehole_A_depths[0]); i++)
    //   borehole_A_depths[i](0,0);
    // 	//borehole_A_depths[i](Tensor<1,dim,double>());
    Names names(input_path);
    std::vector<double> borehole_depths;
    names.get_depths(borehole_depths,
		     "road");
    if ((sizeof borehole_A_depths)/(sizeof borehole_A_depths[0])<borehole_depths.size()/*names.road_depths.size()*/)
      {
	pcout << "Not enough points for borehole depths definition" << std::endl;
	throw 1;
      }
    for (unsigned int i=0; i</*names.road_depths*/borehole_depths.size(); i++)
      {
	borehole_A_depths[i][0]= 14.5;
	borehole_A_depths[i][1]= -1.*borehole_depths[i];/*names.road_depths[i]*/
	    
	borehole_F_depths[i][0]=  0.0;
	borehole_F_depths[i][1]= -1.*borehole_depths[i];/*names.road_depths[i]*/
	      
	borehole_H_depths[i][0]=  2.5;
	borehole_H_depths[i][1]= -1.*borehole_depths[i];/*names.road_depths[i]*/

	borehole_I_depths[i][0]=  5.0;
	borehole_I_depths[i][1]= -1.*borehole_depths[i];/*names.road_depths[i]*/	    
      }
  }
      
  // template<int dim>
  // const Point<dim> 
  // Heat_Pipe<dim>::stores_centers[2]
  // = { Point<dim> (0.,-1.*collector_depth),
  //     Point<dim> (0.,-1.*storage_depth)};
  
  template<int dim>
  Heat_Pipe<dim>::~Heat_Pipe ()
  {
    dof_handler_temperature.clear ();
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  void Heat_Pipe<dim>::read_mesh_temperature()
  {
    std::string filename;
    if (dim==3)
      filename=input_path+"/meshes/trl_final_4_merged.msh";
    else if (dim==2)
      filename=input_path+"/meshes/trl_mesh_in_2d.msh";
    else
      {
	pcout << "Error in read_mesh_temperature. "
		  << "Case not implemented." << std::endl;
	throw -1;
      }
	  
    DataTools data_tools;
    if (data_tools.file_exists(filename)==false)
      {
	pcout << "Error file: " << filename << " does not exists!"
		  << std::endl;
	throw -1;
      }
    else
      {
	pcout << "Reading mesh from: " << filename << std::endl;
      }
    
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation_temperature);
    std::ifstream file (filename.c_str());
    grid_in.read_msh (file);
  }
//   //**************************************************************************************
//   //--------------------------------------------------------------------------------------
//   //**************************************************************************************
//   template <int dim>
//   void Heat_Pipe<dim>::refine_grid ()
//   {
//     // const PETScWrappers::Vector localized_old_solution (backup_solution_temperature);
//     // Vector<float> local_error_per_cell (triangulation_temperature.n_active_cells()); 
    
//     // KellyErrorEstimator<dim>::estimate (dof_handler_temperature,
//     // 					QGauss<dim-1>(3),
//     // 					typename FunctionMap<dim>::type(),
//     // 					localized_old_solution,
//     // 					local_error_per_cell,
//     // 					ComponentMask(),
//     // 					0,
//     // 					multithread_info.n_default_threads,
//     // 					this_mpi_process);
    
//     // const unsigned int n_local_cells
//     //   = GridTools::count_cells_with_subdomain_association (triangulation_temperature,
//     // 							   this_mpi_process);
//     // PETScWrappers::MPI::Vector
//     //   distributed_all_errors (mpi_communicator,
//     // 			      triangulation_temperature.n_active_cells(),
//     // 			      n_local_cells);
    
//     // for (unsigned int i=0; i<local_error_per_cell.size(); ++i)
//     //   if (local_error_per_cell(i)!=0)
//     // 	distributed_all_errors(i)=local_error_per_cell(i);
//     // distributed_all_errors.compress(VectorOperation::insert);
    
//     // local_error_per_cell=distributed_all_errors;
//     /*--------------------------------------------------------------------*/
//     /*
//       Refine grid adaptatively based on the previous estimated errors
//     */
//     SolutionTransfer< dim,PETScWrappers::Vector/*,DoFHandler<dim>*/ > 
//       solution_transfer(dof_handler_temperature);
    
//     std::vector<PETScWrappers::Vector> transfer_in(2);
//     transfer_in[0]=old_solution_temperature;
//     transfer_in[1]=    solution_temperature;
    
//     // GridRefinement::refine_and_coarsen_fixed_number (triangulation_temperature,
//     // 						     local_error_per_cell,
//     // 						     0.35, 0.03);
//     typename Triangulation<dim>::active_cell_iterator
//       cell = triangulation_temperature.begin_active(),
//       endc = triangulation_temperature.end();
//     for (; cell!=endc; ++cell)
//       {
// 	if ((cell->material_id()==10)||
// 	    (cell->material_id()==11)||
// 	    (cell->material_id()==12)||
// 	    (cell->material_id()==13))
// 	  cell->set_refine_flag();
//       }
    
//     triangulation_temperature
//       .prepare_coarsening_and_refinement();
//     solution_transfer
//       .prepare_for_coarsening_and_refinement (transfer_in);
    
//     triangulation_temperature.execute_coarsening_and_refinement ();
    
//     setup_system_temperature ();
	  
//     std::vector<PETScWrappers::Vector> transfer_out (2); 
//     transfer_out[0].reinit(dof_handler_temperature.n_dofs());
//     transfer_out[1].reinit(dof_handler_temperature.n_dofs());
	  
//     solution_transfer.interpolate (transfer_in, transfer_out);
    
//     old_solution_temperature=transfer_out[0];
//     solution_temperature=transfer_out[1];
//   }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  void Heat_Pipe<dim>::setup_system_temperature()
  {
    GridTools::partition_triangulation(n_mpi_processes,triangulation_temperature);
    dof_handler_temperature.distribute_dofs (*fe_temperature);
    DoFRenumbering::subdomain_wise (dof_handler_temperature);
    
    local_dofs_per_process_temperature.resize(n_mpi_processes);
    
    for (unsigned int i=0; i<n_mpi_processes; ++i)
      local_dofs_per_process_temperature[i]
	=DoFTools::count_dofs_with_subdomain_association (dof_handler_temperature, i);
    
    n_local_dofs_temperature = local_dofs_per_process_temperature[this_mpi_process];
    
    constraints_temperature.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_temperature,constraints_temperature);
    constraints_temperature.close();
    
    DynamicSparsityPattern sparsity_pattern(dof_handler_temperature.n_dofs(),
					    dof_handler_temperature.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_temperature,sparsity_pattern);
    constraints_temperature.condense(sparsity_pattern);

    old_solution_temperature
      .reinit(mpi_communicator,
	      dof_handler_temperature.n_dofs(),
	      n_local_dofs_temperature);
    backup_solution_temperature
      .reinit(mpi_communicator,
	       dof_handler_temperature.n_dofs(),
	       n_local_dofs_temperature);
    solution_temperature
      .reinit(mpi_communicator,
	      dof_handler_temperature.n_dofs(),
	      n_local_dofs_temperature);
    system_rhs_temperature
      .reinit(mpi_communicator,
	      dof_handler_temperature.n_dofs(),
	      n_local_dofs_temperature);
    system_matrix_temperature
      .reinit(mpi_communicator,
	      sparsity_pattern,
	      local_dofs_per_process_temperature,
	      local_dofs_per_process_temperature,
	      this_mpi_process);
    mass_matrix_temperature
      .reinit(mpi_communicator,
	      sparsity_pattern,
	      local_dofs_per_process_temperature,
	      local_dofs_per_process_temperature,
	      this_mpi_process);
    laplace_matrix_new_temperature
      .reinit(mpi_communicator,
	      sparsity_pattern,
	      local_dofs_per_process_temperature,
	      local_dofs_per_process_temperature,
	      this_mpi_process);
    laplace_matrix_old_temperature
      .reinit(mpi_communicator,
	      sparsity_pattern,
	      local_dofs_per_process_temperature,
	      local_dofs_per_process_temperature,
	      this_mpi_process);
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  void Heat_Pipe<dim>::assemble_system_petsc_temperature()
  {    
    PETScWrappers::MPI::Vector tmp;
    tmp.reinit       (mpi_communicator,
    		      dof_handler_temperature.n_dofs (),
    		      n_local_dofs_temperature);

    mass_matrix_temperature.vmult(tmp,old_solution_temperature);
    tmp.compress(VectorOperation::insert);

    system_rhs_temperature+=tmp;
    system_rhs_temperature.compress(VectorOperation::add);

    laplace_matrix_old_temperature.vmult(tmp,old_solution_temperature);
    tmp.compress(VectorOperation::insert);

    system_rhs_temperature.add(-(1-theta_temperature)*time_step,tmp);
    system_rhs_temperature.compress(VectorOperation::add);
 
    system_matrix_temperature = 0.;
    system_matrix_temperature.compress(VectorOperation::insert);
    system_matrix_temperature.add(1.0,mass_matrix_temperature);
    system_matrix_temperature.compress(VectorOperation::add);

    system_matrix_temperature.add(theta_temperature*time_step,laplace_matrix_new_temperature);
    system_matrix_temperature.compress(VectorOperation::add);

    if (fixed_bc_at_bottom)
      {
	std::map<unsigned int,double> boundary_values;
	VectorTools::interpolate_boundary_values (dof_handler_temperature,
						  boundary_id_soil_bottom,
						  ConstantFunction<dim>(10.95),
						  boundary_values);
	MatrixTools::apply_boundary_values (boundary_values,
					    system_matrix_temperature,
					    solution_temperature,
					    system_rhs_temperature,
					    false);
      }
    if (author=="Fixed")
      {
	std::map<unsigned int,double> boundary_values;
	VectorTools::interpolate_boundary_values (dof_handler_temperature,
						  boundary_id_road,
						  ConstantFunction<dim>(new_air_temperature),
						  boundary_values);
	MatrixTools::apply_boundary_values (boundary_values,
					    system_matrix_temperature,
					    solution_temperature,
					    system_rhs_temperature,
					    false);
	boundary_values.clear();
	VectorTools::interpolate_boundary_values (dof_handler_temperature,
						  boundary_id_soil,
						  ConstantFunction<dim>(new_air_temperature),
						  boundary_values);
	MatrixTools::apply_boundary_values (boundary_values,
					    system_matrix_temperature,
					    solution_temperature,
					    system_rhs_temperature,
					    false);
      }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  void Heat_Pipe<dim>::assemble_system_parallel_temperature(/*unsigned int step,*/
							    bool system_switch)
  {
    mass_matrix_temperature       =0.;
    laplace_matrix_new_temperature=0.;
    laplace_matrix_old_temperature=0.;
    system_rhs_temperature        =0.;
    mass_matrix_temperature.compress       (VectorOperation::insert);
    laplace_matrix_new_temperature.compress(VectorOperation::insert);
    laplace_matrix_old_temperature.compress(VectorOperation::insert);
    system_rhs_temperature.compress        (VectorOperation::insert);
    
    const QGauss<dim> quadrature_formula(3);
    const QGauss<dim-1> face_quadrature_formula(3);
    FEValues<dim> fe_values(*fe_temperature, quadrature_formula,
			    update_values | update_gradients |
			    update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(*fe_temperature, face_quadrature_formula,
				     update_values | update_gradients |
				     update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell=fe_temperature->dofs_per_cell;
    const unsigned int n_q_points=quadrature_formula.size(); 
    const unsigned int n_face_q_points=face_quadrature_formula.size ();
   
    FullMatrix<double> cell_mass_matrix       (dofs_per_cell,dofs_per_cell);
    FullMatrix<double> cell_laplace_matrix_new(dofs_per_cell,dofs_per_cell);
    FullMatrix<double> cell_laplace_matrix_old(dofs_per_cell,dofs_per_cell);
    Vector<double>     cell_rhs               (dofs_per_cell);
    
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    
    double face_boundary_indicator;
        
    unsigned int faces_on_road_surface = 0;
    unsigned int faces_on_soil_surface = 0;

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_temperature.begin_active(),
      endc = dof_handler_temperature.end();
    for (; cell!=endc; ++cell)
      if (cell->subdomain_id()==this_mpi_process)
	{
	  fe_values.reinit(cell);

	  cell_mass_matrix=0;
	  cell_laplace_matrix_new=0;
	  cell_laplace_matrix_old=0;
	  cell_rhs=0;
	  
	  MaterialData material_data(dim,insulation,0,moisture_movement);
	  double thermal_conductivity=material_data.get_soil_thermal_conductivity(cell->material_id());
	  double thermal_heat_capacity=material_data.get_soil_heat_capacity(cell->material_id());
	  double density=material_data.get_soil_density(cell->material_id());
	  
	  /*
	    studying the impact of thermal conductivity variations under
	    the insulation layer due (maybe) to the installation process
	  */
	  if ((cell->center()[0]< 6.0) &&   // insulation edge
	      (cell->center()[0]>-6.0) &&   // insulation edge
	      (cell->center()[1]<-0.725) && // insulation depth
	      (cell->center()[1]>-9.0) &&   // assumed thermal penetration
	      (preheating_step>=4)&&
	      cell->material_id()==14)
	    thermal_conductivity=thermal_conductivity*thermal_conductivity_factor;
	  
	  for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	    for (unsigned int i=0; i<dofs_per_cell; ++i)
	      for (unsigned int j=0; j<dofs_per_cell; ++j)
		{
		  cell_mass_matrix(i,j)+=(thermal_heat_capacity*density*
					  fe_values.shape_value(i,q_point)*
					  fe_values.shape_value(j,q_point)*
					  fe_values.JxW(q_point));
		  cell_laplace_matrix_new(i,j)+=(thermal_conductivity*
						 fe_values.shape_grad(i,q_point) *
						 fe_values.shape_grad(j,q_point) *
						 fe_values.JxW(q_point));
		  cell_laplace_matrix_old(i,j)+=(thermal_conductivity * 
						 fe_values.shape_grad(i,q_point) *
						 fe_values.shape_grad(j,q_point) *
						 fe_values.JxW(q_point));
		}
	  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	    {                                                                               
	      face_boundary_indicator = cell->face(face)->boundary_id();
	      /*
		We enter this section if we found a cell face that represents a boundary.
		Then we will calculate the corresponding heat flux according to the 
		boundary we found. For the case of the pipe boundaries, it is also
		necessary for the pipe system to be active. If it isn't then we do nothing
		for these cells. This would correspond to a homogeneous boundary condition
		of second kind (i.e. insulation)
	      */
	      if ((cell->face(face)->at_boundary()) &&
		  (((author!="Fixed") &&
		    ((face_boundary_indicator==boundary_id_road) ||
		     (face_boundary_indicator==boundary_id_soil))) ||
		   ((pipe_system==true) && (system_switch==true) &&
		    ((face_boundary_indicator==boundary_id_collector) ||
		     (face_boundary_indicator==boundary_id_storage  )))))
		{
		  /*
		    Variables for heat flux from surface. The convective coefficients below
		    include both convective and infrared interactions since they can be
		    represented in a similar way (after the radiative coefficients have
		    been linearized).
		  */
		  //double inbound_convective_coefficient_new=0;
		  //double inbound_convective_coefficient_old=0;
		  double outbound_convective_coefficient_new=0;
		  double outbound_convective_coefficient_old=0;
		  double inbound_heat_flux_new=0.;
		  double inbound_heat_flux_old=0.;
		  
		  if ((face_boundary_indicator==boundary_id_road) ||
		      (face_boundary_indicator==boundary_id_soil))
		    {
		      /*
			If 'analytic == true' then the remaining variables passed to the object of
			class type 'BoundaryConditions' are irrelevant because they are redefined 
			when its flux members are called according to the 'time' provided.
		    
			Some of the heat transfer coefficients (infrared coefficients in particular)
			depend on an estimation of the previous surface temperature. At the beggining
			i thought it would be ok to just ask the point_value at each cell. The problem
			is that this operation is highly costly (as it needs to iterate and interpolate
			in the solution vector) and the cost is directly proportional to the number of
			cells (making it unsuitable for 3D). So, a new strategy is needed.
		      */
		      double old_surface_temperature=-1000.;
		      if (face_boundary_indicator==boundary_id_road)
		      	old_surface_temperature=old_avg_road_surface_temperature;
		      else if (face_boundary_indicator==boundary_id_soil)
		      	old_surface_temperature=old_avg_soil_surface_temperature;
		      
		      double new_surface_temperature=-1000.;
		      if (face_boundary_indicator==boundary_id_road)
			{
			  new_surface_temperature
			    =0.5*current_new_avg_road_surface_temperature
			    +0.5*previous_new_road_avg_surface_temperature;
			}
		      else if (face_boundary_indicator==boundary_id_soil)
			{
			  new_surface_temperature
			    =0.5*current_new_avg_soil_surface_temperature
			    +0.5*previous_new_soil_avg_surface_temperature;
			}
		      else
			{
			  if (cell_index_to_previous_new_surface_temperature.find(cell)!=
			      cell_index_to_previous_new_surface_temperature.end())
			    new_surface_temperature
			      =0.5*cell_index_to_previous_new_surface_temperature[cell]
			      +0.5*cell_index_to_current__new_surface_temperature[cell];
			  else
			    {
			      std::cout << "Error, new cell value  not found. Cell index: " << cell << "\t"
					<< "MPI process: " << this_mpi_process << std::endl;
			      throw -1;
			    }
			}
		      		      
		      BoundaryConditions boundary_condition_old (false/*analytic*/,
								 time_step*(timestep_number-1),
								 old_air_temperature,
								 old_solar_radiation,
								 old_wind_speed,
								 old_relative_humidity,
								 old_precipitation,
								 old_surface_temperature,
								 0./*,old_surface_pressure*/,
								 moisture_movement);
		      BoundaryConditions boundary_condition_new (false/*analytic*/,
								 time_step*timestep_number,
								 new_air_temperature,
								 new_solar_radiation,
								 new_wind_speed,
								 new_relative_humidity,
								 new_precipitation,
								 new_surface_temperature,
								 0./*,new_surface_pressure*/,
								 moisture_movement);
		      double shading_factor=0.;
		      if (shading &&
			  ((face_boundary_indicator==boundary_id_road) &&
			   (((date_and_time[timestep_number][3]>=12) && (date_and_time[timestep_number][3]<=16)) ||
			    ((date_and_time[timestep_number][3]==17) && (date_and_time[timestep_number][4]==00)))))
			shading_factor=shading_factor_value;
		      
		      std::string local_author=author;
		      std::string local_surface_type=surface_type;
		      if ((dim==2 || dim==3) && 
			  (face_boundary_indicator==boundary_id_road))
			{
			  local_author="Herb";
			  local_surface_type="road";
			}
		      
		      double new_canopy_temperature=0.;
		      double old_canopy_temperature=0.;
		      if (local_author=="Best")
			{
			  new_canopy_temperature = 
			    boundary_condition_new
			    .get_canopy_temperature (local_surface_type,
						     local_author,
						     canopy_density);
			  old_canopy_temperature = 
			    boundary_condition_old
			    .get_canopy_temperature (local_surface_type,
						     local_author,
						     canopy_density);
			}
		      /* Heat flux from soil surface or road surface*/
		      if ((local_author=="Herb"   ) || 
			  (local_author=="Jansson") ||
			  (local_author=="Best"   )) 
			{
			  outbound_convective_coefficient_new
			    =boundary_condition_new
			    .get_outbound_coefficient(local_surface_type,
						      local_author,
						      canopy_density,
						      old_surface_temperature,
						      new_surface_temperature,
						      true);
			  outbound_convective_coefficient_old
			    =boundary_condition_old
			    .get_outbound_coefficient(local_surface_type,
						      local_author,
						      canopy_density,
						      old_surface_temperature,
						      new_surface_temperature,
						      false);
			  inbound_heat_flux_new
			    =boundary_condition_new
			    .get_inbound_heat_flux(local_surface_type,
						   local_author,
						   shading_factor,
						   new_canopy_temperature,
						   canopy_density,
						   old_surface_temperature,
						   new_surface_temperature,
						   true);
			  inbound_heat_flux_old
			    =boundary_condition_old
			    .get_inbound_heat_flux(local_surface_type,
						   local_author,
						   shading_factor,
						   old_canopy_temperature,
						   canopy_density,
						   old_surface_temperature,
						   new_surface_temperature,
						   false);
			  /*
			    Save the calculated heat fluxes.
			  */
			  {
			    double solar_heat_flux;
			    double inbound_convective_heat_flux;
			    double inbound_evaporative_heat_flux;
			    double inbound_infrared_heat_flux;
			    double outbound_convective_coefficient;
			    double outbound_infrared_coefficient;
			    double outbound_evaporative_coefficient;
			    boundary_condition_new.print_inbound_heat_fluxes (solar_heat_flux,
									      inbound_convective_heat_flux,
									      inbound_evaporative_heat_flux,
									      inbound_infrared_heat_flux,
									      outbound_convective_coefficient,
									      outbound_infrared_coefficient,
									      outbound_evaporative_coefficient);
			    if (face_boundary_indicator==boundary_id_road)
			      {
				road_heat_fluxes[timestep_number-1][0]+=solar_heat_flux;
				road_heat_fluxes[timestep_number-1][1]+=inbound_convective_heat_flux;
				road_heat_fluxes[timestep_number-1][2]+=inbound_evaporative_heat_flux;
				road_heat_fluxes[timestep_number-1][3]+=inbound_infrared_heat_flux;
				road_heat_fluxes[timestep_number-1][4]+=outbound_convective_coefficient*new_surface_temperature;
				road_heat_fluxes[timestep_number-1][5]+=outbound_infrared_coefficient*new_surface_temperature;
				road_heat_fluxes[timestep_number-1][6]+=outbound_convective_coefficient;
				road_heat_fluxes[timestep_number-1][7]+=outbound_infrared_coefficient;
				road_heat_fluxes[timestep_number-1][8]+=outbound_convective_coefficient_new*new_surface_temperature;
				road_heat_fluxes[timestep_number-1][9]+=inbound_heat_flux_new;
				faces_on_road_surface++;
			      }
			    else if (face_boundary_indicator==boundary_id_soil)
			      {
				soil_heat_fluxes[timestep_number-1][0]+=solar_heat_flux;
				soil_heat_fluxes[timestep_number-1][1]+=inbound_convective_heat_flux;
				soil_heat_fluxes[timestep_number-1][2]+=inbound_evaporative_heat_flux;
				soil_heat_fluxes[timestep_number-1][3]+=inbound_infrared_heat_flux;
				soil_heat_fluxes[timestep_number-1][4]+=outbound_convective_coefficient*new_surface_temperature;
				soil_heat_fluxes[timestep_number-1][5]+=outbound_infrared_coefficient*new_surface_temperature;
				soil_heat_fluxes[timestep_number-1][6]+=outbound_convective_coefficient;
				soil_heat_fluxes[timestep_number-1][7]+=outbound_infrared_coefficient;
				soil_heat_fluxes[timestep_number-1][8]+=outbound_convective_coefficient_new*new_surface_temperature;
				soil_heat_fluxes[timestep_number-1][9]+=inbound_heat_flux_new;
				faces_on_soil_surface++;
			      }
			    else
			      {
				pcout << "Error. face_boundary_id " << face_boundary_indicator << " not implemented.";
				pcout << std::endl;
				throw 1;
			      }
			    
			  }
			}
		    }
		  // collector and storage pipes in 2D
		  else if ((face_boundary_indicator==boundary_id_collector) ||
			   (face_boundary_indicator==boundary_id_storage))
		    {
		      /*
			The heat fluxes are multiplied by -1. because these are
			calculated from the point of view of the pipes, so, if
			we have a positive value this means a loss of energy from
			the soil
		      */
		      unsigned int pipe_number=cell_index_to_pipe_number[cell];
		      if (face_boundary_indicator==boundary_id_collector)
			{
			  inbound_heat_flux_old=-1.*old_pipe_heat_flux[pipe_number]/2.59666;//(2.17);
			  // inbound_heat_flux_new=-1.*(0.5*previous_new_pipe_heat_flux[pipe_number]+
			  // 			     0.5*current__new_pipe_heat_flux[pipe_number])/(2.17);
			  inbound_heat_flux_new=-1.*current__new_pipe_heat_flux[pipe_number]/2.59666;//(2.17);
			}
		      else
			{
			  inbound_heat_flux_old=-1.*old_pipe_heat_flux[pipe_number]/2.59666;//(2.17);//(1.922);
			  // inbound_heat_flux_new=-1.*(0.5*previous_new_pipe_heat_flux[pipe_number]+
			  // 			     0.5*current__new_pipe_heat_flux[pipe_number])/(2.17);
			  inbound_heat_flux_new=-1.*current__new_pipe_heat_flux[pipe_number]/2.59666;//(2.17);
			}
		    }
		  else
		    {
		      pcout << "Error: author not implemented." << std::endl
			    << "Error in assembling function."  << std::endl;
		      throw 3;
		    }

		  fe_face_values.reinit (cell,face);
		  for (unsigned int q_face_point = 0; q_face_point < n_face_q_points; ++q_face_point)
		    for (unsigned int i=0; i<dofs_per_cell; ++i)
		      {
			if ((face_boundary_indicator==boundary_id_road) ||
			    (face_boundary_indicator==boundary_id_soil))
			  for (unsigned int j=0; j<dofs_per_cell; ++j)
			    {
			      cell_laplace_matrix_new (i,j)+=(outbound_convective_coefficient_new * 
							      fe_face_values.shape_value (i,q_face_point) *
							      fe_face_values.shape_value (j,q_face_point) *
							      fe_face_values.JxW         (q_face_point));
			      cell_laplace_matrix_old (i,j)+=(outbound_convective_coefficient_old * 
							      fe_face_values.shape_value (i,q_face_point) *
							      fe_face_values.shape_value (j,q_face_point) *
							      fe_face_values.JxW         (q_face_point));
			    }
			cell_rhs (i)+=((inbound_heat_flux_old *
					time_step * theta_temperature * 
					fe_face_values.shape_value (i,q_face_point) *
					fe_face_values.JxW (q_face_point))
				       +
				       (inbound_heat_flux_new * 
					time_step * (1-theta_temperature) *
					fe_face_values.shape_value (i,q_face_point) *
					fe_face_values.JxW (q_face_point)));
		      }
		}
	    }
	  cell->get_dof_indices (local_dof_indices);
	  constraints_temperature
	    .distribute_local_to_global(cell_laplace_matrix_new,
					local_dof_indices,
					laplace_matrix_new_temperature);
	  constraints_temperature
	    .distribute_local_to_global(cell_laplace_matrix_old,
					local_dof_indices,
					laplace_matrix_old_temperature);
	  constraints_temperature
	    .distribute_local_to_global(cell_mass_matrix,
					cell_rhs,
					local_dof_indices,
					mass_matrix_temperature, system_rhs_temperature);
	}
    laplace_matrix_new_temperature.compress(VectorOperation::add);
    laplace_matrix_old_temperature.compress(VectorOperation::add);
    mass_matrix_temperature.compress(VectorOperation::add);
    system_rhs_temperature.compress(VectorOperation::add);
    
    for (unsigned int i=0; i<road_heat_fluxes[timestep_number-1].size(); i++)
      {
	road_heat_fluxes[timestep_number-1][i]
	  =Utilities::MPI::sum(road_heat_fluxes[timestep_number-1][i],mpi_communicator)
	  /Utilities::MPI::sum(faces_on_road_surface,mpi_communicator);
	  
	soil_heat_fluxes[timestep_number-1][i]
	  =Utilities::MPI::sum(soil_heat_fluxes[timestep_number-1][i],mpi_communicator)
	  /Utilities::MPI::sum(faces_on_soil_surface,mpi_communicator);
      }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  unsigned int Heat_Pipe<dim>::solve_temperature()
  {
    SolverControl solver_control(solution_temperature.size(),
				 1e-8*system_rhs_temperature.l2_norm());
    PETScWrappers::SolverCG solver(solver_control,
				   mpi_communicator);
    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix_temperature);
    solver.solve(system_matrix_temperature,solution_temperature,
		 system_rhs_temperature,preconditioner);
    Vector<double> localized_solution(solution_temperature);
    constraints_temperature.distribute(localized_solution);
    solution_temperature=localized_solution;
    solution_temperature.compress(VectorOperation::insert);
    return solver_control.last_step();
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  void Heat_Pipe<dim>::output_results()
  {
    const Vector<double> localized_solution (solution_temperature);

    if (this_mpi_process==0)
      {
	std::string dirpath="./output";
	DataTools data_tools;
	if (data_tools.dir_exists(dirpath)==false)
	  {
	    std::cout << "Error directory: " << dirpath << " does not exists!"
		      << std::endl;
	    throw -1;
	  }
	else
	  {
	    std::cout << "Writing output to: " << dirpath << std::endl;
	  }


	std::stringstream t;
	t << timestep_number;
	std::string filename = "./output/solution_" + preheating_output_filename
	  +"_timestep-" + t.str();

	DataOut<dim> data_out;
	data_out.attach_dof_handler (dof_handler_temperature);

	std::vector<std::string> solution_names;
	solution_names.push_back ("temperature");

	data_out.add_data_vector (localized_solution,solution_names);
	/*
	 * Add information about in which mpi process is each cell being processed
	 * */
	std::vector<unsigned int>
	  partition_int(triangulation_temperature.n_active_cells());
	GridTools::get_subdomain_association
	  (triangulation_temperature, partition_int);
	const Vector<double> partitioning (partition_int.begin (),
					   partition_int.end ());
	data_out.add_data_vector (partitioning, "partitioning");
	/*
	 * Add information about whats the material on each cell
	 * and thermal properties
	 * */
	MaterialData material_data (dim,insulation,/*moisture content*/0.23,moisture_movement);
	std::vector<unsigned int> material_id_int;
	std::vector<double> thermal_conductivity_int;
	std::vector<double> specific_heat_capacity_int;
	std::vector<double> density_int;
	std::vector<double> thermal_diffusivity_int;

	std::vector<unsigned int> boundaries;
	typename DoFHandler<dim>::active_cell_iterator
	  cell = dof_handler_temperature.begin_active(),
	  endc = dof_handler_temperature.end();
	for (; cell!=endc; ++cell)
	  {
	    material_id_int.push_back(cell->material_id());

	    if ((cell->center()[0]< 6.0) &&   // insulation edge
		(cell->center()[0]>-6.0) &&   // insulation edge
		(cell->center()[1]<-0.725) && // insulation depth
		(cell->center()[1]>-9.0) &&   // assumed thermal penetration
		(preheating_step>=4) &&
		cell->material_id()==14) // soil material id
	      thermal_conductivity_int.push_back(thermal_conductivity_factor*
						 material_data.get_soil_thermal_conductivity(cell->material_id()));
	    else
	      thermal_conductivity_int.push_back(material_data.get_soil_thermal_conductivity(cell->material_id()));

	    //thermal_conductivity_int.push_back  (material_data.get_soil_thermal_conductivity(cell->material_id()));
	    specific_heat_capacity_int.push_back(material_data.get_soil_heat_capacity       (cell->material_id()));
	    density_int.push_back               (material_data.get_soil_density             (cell->material_id()));
	    thermal_diffusivity_int.push_back   (material_data.get_soil_thermal_diffusivity (cell->material_id()));

	    if (cell_index_to_previous_new_surface_temperature.find(cell)!=
		cell_index_to_previous_new_surface_temperature.end())
	      boundaries.push_back(cell->face(cell_index_to_face_index[cell])->boundary_id());
	    else
	      boundaries.push_back(0);
	  }
	const Vector<double> material_id           (material_id_int.begin(),
						    material_id_int.end());
	const Vector<double> thermal_conductivity  (thermal_conductivity_int.begin(),
						    thermal_conductivity_int.end());
	const Vector<double> specific_heat_capacity(specific_heat_capacity_int.begin(),
						    specific_heat_capacity_int.end());
	const Vector<double> density               (density_int.begin(),
						    density_int.end());
	const Vector<double> thermal_diffusivity   (thermal_diffusivity_int.begin(),
						    thermal_diffusivity_int.end());
	const Vector<double> boundary_id           (boundaries.begin(),
						    boundaries.end());

	data_out.add_data_vector (material_id           ,"material_id");
	data_out.add_data_vector (thermal_conductivity  ,"thermal_conductivity");
	data_out.add_data_vector (specific_heat_capacity,"specific_heat_capacity");
	data_out.add_data_vector (density               ,"density");
	data_out.add_data_vector (thermal_diffusivity   ,"thermal_diffusivity");
	data_out.add_data_vector (boundary_id   ,"boundary_id");

	data_out.build_patches ();
	if (dim==1)
	  filename = filename + ".gp";
	if (dim==3 || dim==2)
	  filename = filename + ".vtu";
	std::ofstream output (filename.c_str());
	if (dim==1)
	  data_out.write_gnuplot (output);
	if (dim==3 || dim==2)
	  data_out.write_vtu (output);
      }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  void Heat_Pipe<dim>::fill_output_vectors()
  {
    const Vector<double> localized_solution_temperature(old_solution_temperature);

    Names names(input_path);
    std::vector<double>borehole_depths;
    names.get_depths(borehole_depths,
		     "road");
    unsigned int soil_depths=borehole_depths.size();//names.road_depths.size();
    //std::vector<double> depths_row(soil_depths,0);
    std::vector<double> soil_bha_temperature_row(soil_depths,0.);
    std::vector<double> soil_bhf_temperature_row(soil_depths,0.);
    std::vector<double> soil_bhh_temperature_row(soil_depths,0.);
    std::vector<double> soil_bhi_temperature_row(soil_depths,0.);
    
    unsigned int index=0;
    unsigned int depths=0;
    for (unsigned int i=0; i<5*soil_depths; i++)
      {
	if (i>=n_mpi_processes*(index+1))
	  index++;
	if (i>=soil_depths*(depths+1))
	  depths++;

	// if (timestep_number==1 && index==0)
	//   depths_row[i]
	//     =borehole_depths[i];

	if (this_mpi_process==(i-index*n_mpi_processes))
	  {
	    if(i<1*soil_depths)
	      soil_bha_temperature_row[i-depths*soil_depths]
		=VectorTools::point_value(dof_handler_temperature, localized_solution_temperature, 
					   borehole_A_depths[i-soil_depths*depths]);
	    if(i<2*soil_depths && i>=1*soil_depths)
	      soil_bhf_temperature_row[i-depths*soil_depths]
		=VectorTools::point_value(dof_handler_temperature, localized_solution_temperature,
					  borehole_F_depths[i-soil_depths*depths]);
	    if(i<3*soil_depths && i>=2*soil_depths)
	      soil_bhh_temperature_row[i-depths*soil_depths]
		=VectorTools::point_value(dof_handler_temperature, localized_solution_temperature,
					  borehole_H_depths[i-soil_depths*depths]);
	    if(i<4*soil_depths && i>=3*soil_depths)
	      soil_bhi_temperature_row[i-depths*soil_depths]
		=VectorTools::point_value(dof_handler_temperature, localized_solution_temperature,
					  borehole_I_depths[i-soil_depths*depths]);
	  }
      }
    std::vector< std::vector<double> > temp;
    temp.push_back(soil_bha_temperature_row);
    temp.push_back(soil_bhf_temperature_row);
    temp.push_back(soil_bhh_temperature_row);
    temp.push_back(soil_bhi_temperature_row);

    // if (timestep_number==1)
    //   {
    // 	soil_bha_temperature.push_back(depths_row);
    // 	soil_bhf_temperature.push_back(depths_row);
    // 	soil_bhh_temperature.push_back(depths_row);
    // 	soil_bhi_temperature.push_back(depths_row);
    //   }
    for (unsigned int i=0; i<temp.size(); i++)
      {
	std::vector<double> temp_row;
	for (unsigned int j=0; j<temp[i].size(); j++)
	  temp_row.push_back(Utilities::MPI::sum(temp[i][j],mpi_communicator));
	if (i==0)
	  soil_bha_temperature.push_back(temp_row);
	if (i==1)
	  soil_bhf_temperature.push_back(temp_row);
	if (i==2)
	  soil_bhh_temperature.push_back(temp_row);
	if (i==3)
	  soil_bhi_temperature.push_back(temp_row);
      }
    if (pipe_system==true)
      pipe_heat_fluxes.push_back(current__new_pipe_heat_flux);
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  void Heat_Pipe<dim>::mesh_info()
  {
    /*
      Clear all the maps that we need to store the cell data
    */
    cell_index_to_face_index.clear();
    cell_index_to_mpi_process.clear();
    cell_index_to_pipe_number.clear();
    cell_index_to_previous_new_surface_temperature.clear();
    cell_index_to_current__new_surface_temperature.clear();
    cell_index_to_old_surface_temperature.clear();
    //cell_index_3d_to_old_pipe_heat_flux.clear();

    std::map<unsigned int,unsigned int> boundary_count;
    for (unsigned int i=0; i<n_boundary_ids; i++)
      boundary_count[i]=0;
    int number_of_local_cells = 0;

    unsigned int mpi_index = 0;
    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_temperature.begin_active(),
      endc = dof_handler_temperature.end();
    for (; cell!=endc; ++cell)
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
  	if (cell->face(face)->at_boundary())
  	  {
  	    unsigned int boundary_id
	      =cell->face(face)->boundary_id();
  	    Point<dim> cell_center
	      =cell->face(face)->center();
	    // double coordinate_x
	    //   =cell_center[0];
  	    /*
  	      We shouldn't have any boundary with 0 indicator index. If
  	      we found any, print it.
  	    */
  	    if (cell->subdomain_id()==this_mpi_process)
  	      number_of_local_cells++;
  	    if (boundary_id==0)
  	      pcout << cell_center << std::endl;
  	    boundary_count[boundary_id]++;
	    
  	    if ((boundary_id==boundary_id_road)||
  		(boundary_id==boundary_id_soil)||
  		((pipe_system==true)&&((boundary_id==boundary_id_collector)||
  				       (boundary_id==boundary_id_storage))))
  	      {
		cell_index_to_previous_new_surface_temperature[cell]=18.;
  		cell_index_to_current__new_surface_temperature[cell]=18.;
  		cell_index_to_old_surface_temperature[cell]=18.;
  		cell_index_to_face_index[cell]=face;
		
		if ((boundary_id==boundary_id_collector)||
		    (boundary_id==boundary_id_storage))
		  {
		    double x=cell_center[0];
		    double y=cell_center[1];
		    double cell_radius=0.;
		    if (((x>0. || x<0.) && dim==3 && (y>12. && y<42.)) ||
			((x>0. || x<0.) && dim==2))
		      cell_radius=fabs(x);
		    else if (y>42. && dim==3)
		      cell_radius=sqrt(pow(fabs(x),2)+pow(y-42.,2));
		    else
		      {
			pcout << "Error in cell: " << cell 
			      << " with boundary face center in: " 
			      << cell_center << std::endl;
			throw -1;
		      }
		    
		    for (unsigned int i=0; i<10; i++)
		      {
			double r=0.;
			r=(0.25/2 + (double)i*0.25);
			
			//bool cell_found=false;
			double dr=0.;
			if (dim==3 && y>42.)
			  dr=r*(1.-cos(22.5*(numbers::PI/180.)));
			
			if ((dim==3) &&
			    (cell_radius<((r-dr)+0.03)) &&
			    (cell_radius>((r-dr)-0.03)))
			  {
			    if (boundary_id==boundary_id_collector)
			      cell_index_to_pipe_number[cell]=i;
			    else
			      cell_index_to_pipe_number[cell]=i+10;
			    break;
			  }
			if ((dim==2) &&
			    (cell_radius<((r-dr)+0.03)) &&
			    (cell_radius>((r-dr)-0.03)))
			  {
			    if (boundary_id==boundary_id_collector)
			      {
				if (x>0.)
				  cell_index_to_pipe_number[cell]=10+i;
				else
				  cell_index_to_pipe_number[cell]= 9-i;
			      }
			    else
			      {
				if (x>0.)
				  cell_index_to_pipe_number[cell]=30+i;
				else
				  cell_index_to_pipe_number[cell]=29-i;
			      }
			    break;
			  }
		      }
		  }
  		if (mpi_index>=n_mpi_processes)
  		  mpi_index=0;
  		cell_index_to_mpi_process[cell]=mpi_index;
  		mpi_index++;
  	      }
  	  }
    pcout << "\tNumber of mpi processes: " 
  	  << n_mpi_processes
  	  << std::endl;
    pcout << "\t------Mesh info------" << std::endl
    	  << "\t dimension..........: " << dim << std::endl
    	  << "\t total no. of cells.: " << triangulation_temperature.n_active_cells() << std::endl
  	  << "\t----Boundary info----" << std::endl
  	  << "\t no. of cells (local)..........: " << number_of_local_cells << std::endl
    	  << "\t no. of cells (all mpi process): " 
    	  << Utilities::MPI::sum(number_of_local_cells,mpi_communicator)<< std::endl
  	  << "\t-----Misc. info-----" << std::endl
    	  << "\t size of solution vectors : "
    	  <<     solution_temperature.size() << " (new)" << "\t" 
    	  << old_solution_temperature.size() << " (old)" << std::endl;
    
    pcout << "\tboundary indicators: " << std::endl;
    
    for (std::map<unsigned int, unsigned int>::iterator 
    	   it=boundary_count.begin(); 
    	 it!=boundary_count.end(); ++it)
      pcout << "\t" << it->first 
  	    << "("  << it->second
  	    << " times) "
  	    << std::endl;
    pcout << std::endl << std::endl;

    
    // for (typename std::map<typename DoFHandler<dim>::active_cell_iterator,unsigned int >::iterator 
    // 	   it=cell_index_to_pipe_number.begin(); 
    // 	 it!=cell_index_to_pipe_number.end(); ++it)
    //   pcout << it->first << "\t" 
    // 	    << it->first->face(cell_index_to_face_index[it->first])->center() << "\t"
    // 	    << it->second << "\n";
    // pcout << std::endl;
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  void Heat_Pipe<dim>::surface_temperatures()
  { 
    std::vector<double> local_new_surface_temperatures;
    {
      const QGauss<dim-1> face_quadrature_formula(3);
      const unsigned int  n_face_q_points=face_quadrature_formula.size ();
      std::vector<double> old_face_q_values(n_face_q_points);
      std::vector<double> new_face_q_values(n_face_q_points);
      FEFaceValues<dim>   fe_face_values(*fe_temperature, face_quadrature_formula,
					 update_values | update_gradients |
					 update_quadrature_points | update_JxW_values);
      const Vector<double> localized_new_solution(solution_temperature);
      /*Calculate temperature*/
      for (typename std::map<typename DoFHandler<dim>::active_cell_iterator,unsigned int>::iterator
  	     it=cell_index_to_face_index.begin(); 
  	   it!=cell_index_to_face_index.end(); ++it)
  	{
	  double new_temperature=0.;
  	  if (cell_index_to_mpi_process[it->first]==this_mpi_process)
  	    {
  	      fe_face_values.reinit(it->first,it->second);
	      fe_face_values.get_function_values(localized_new_solution,new_face_q_values);
	      for (unsigned int q_face_point=0; q_face_point<n_face_q_points; q_face_point++)
		new_temperature+=new_face_q_values[q_face_point];/*fe_face_values.JxW(q_face_point);*/
	      new_temperature/=n_face_q_points;
  	    }
  	  local_new_surface_temperatures.push_back(new_temperature);
  	}
    }
    {
      unsigned int vector_index=0;
      cell_index_to_current__new_surface_temperature.clear();
      for (typename std::map<typename DoFHandler<dim>::active_cell_iterator,unsigned int>::iterator
  	     it=cell_index_to_face_index.begin(); 
  	   it!=cell_index_to_face_index.end(); ++it)
	{
    	    cell_index_to_current__new_surface_temperature[it->first]
    	      =Utilities::MPI::sum(local_new_surface_temperatures[vector_index],mpi_communicator);
    	    vector_index++;
	}
    }
    /*
      Finally, calculate the average surface temperature
      AND
      Distribute temperatures around pipes
    */
    {
      for (unsigned int i=0; i<number_of_pipes; i++)
      	{
	  max_pipe_temperature[i]=-1.e6;
      	  new_avg_pipe_temperature[i]=0.;
	  min_pipe_temperature[i]=1.e6;
      	  cell_faces_per_pipe[i]=0;
      	}
      current_new_max_soil_surface_temperature=-1.e6;
      current_new_max_road_surface_temperature=-1.e6;
      current_new_avg_soil_surface_temperature=0.;
      current_new_avg_road_surface_temperature=0.;
      current_new_min_soil_surface_temperature=1.e6;
      current_new_min_road_surface_temperature=1.e6;
      unsigned int cells_on_soil_surface=0;
      unsigned int cells_on_road_surface=0;
      for (typename std::map<typename DoFHandler<dim>::active_cell_iterator,double >::iterator
  	     it=cell_index_to_current__new_surface_temperature.begin(); 
  	   it!=cell_index_to_current__new_surface_temperature.end(); ++it)
  	{
  	  unsigned int face
	    =cell_index_to_face_index[it->first];
  	  unsigned int boundary_id
  	    =it->first->face(face)->boundary_id();

  	  if (boundary_id==boundary_id_soil)
  	    {
	      if (it->second>current_new_max_soil_surface_temperature)
		current_new_max_soil_surface_temperature
		  =it->second;
	      if (it->second<current_new_min_soil_surface_temperature)
		current_new_min_soil_surface_temperature
		  =it->second;
  	      current_new_avg_soil_surface_temperature
		+=it->second;
  	      cells_on_soil_surface++;
  	    }
  	  if (boundary_id==boundary_id_road)
  	    {
	      if (it->second>current_new_max_road_surface_temperature)
		current_new_max_road_surface_temperature
		  =it->second;
	      if (it->second<current_new_min_road_surface_temperature)
		current_new_min_road_surface_temperature
		  =it->second;
  	      current_new_avg_road_surface_temperature
		+=it->second;
  	      cells_on_road_surface++;
  	    }
	  
  	  if ((boundary_id==boundary_id_collector)||
  	      (boundary_id==boundary_id_storage))
  	    {
	      unsigned int pipe_number=cell_index_to_pipe_number.find(it->first)->second; //0 to 39 in 2D
	      if (it->second>max_pipe_temperature[pipe_number])
		max_pipe_temperature[pipe_number]
		  =it->second;
	      if (it->second<min_pipe_temperature[pipe_number])
		min_pipe_temperature[pipe_number]
		  =it->second;
	      new_avg_pipe_temperature[pipe_number]+=it->second;
	      cell_faces_per_pipe[pipe_number]++;
	    }
  	}
      current_new_avg_soil_surface_temperature/=cells_on_soil_surface;
      current_new_avg_road_surface_temperature/=cells_on_road_surface;
      for (unsigned int i=0; i<number_of_pipes; i++)
	new_avg_pipe_temperature[i]
	  /=cell_faces_per_pipe[i];
    }
    // for (typename std::map<typename DoFHandler<dim>::active_cell_iterator,double >::iterator
    // 	   it=cell_index_to_current__new_surface_temperature.begin(); 
    // 	 it!=cell_index_to_current__new_surface_temperature.end(); ++it)
    //   {
    // 	pcout<<it->second<<"\t"
    // 	     <<cell_index_to_previous_new_surface_temperature.find(it->first)->second<<"\t"
    // 	     <<cell_index_to_old_surface_temperature.find(it->first)->second<<"\n";
    //   }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  void Heat_Pipe<dim>::update_met_data()
  {
    if (date_and_time.size()==0 &&
	met_data.size()     ==0)
      {
	if (analytic_met_data)
	  {
	    std::vector< std::vector<int> > initial_date_wrapper;
	    initial_date_wrapper.push_back(initial_date);
	    
	    std::vector< std::vector<int> > initial_date_in_seconds;

	    DataTools data_tools;
	    data_tools.date_to_seconds(initial_date_wrapper,
				       initial_date_in_seconds);
	    std::vector< std::vector<int> > all_dates_in_seconds;
	    for (unsigned int i=0; i<=timestep_number_max; i++)
	      {
		std::vector<int> all_dates_in_seconds_line;

		all_dates_in_seconds_line.push_back(initial_date_in_seconds[0][0] +
						    ((int)time_step*i));

		all_dates_in_seconds.push_back(all_dates_in_seconds_line);
	      }

	    data_tools.seconds_to_date(date_and_time,
				       all_dates_in_seconds);
	    
	    AnalyticSolution analytic_solution(0,0,0,"",false,false,type_of_weather);
	    for (unsigned int i=0; i<date_and_time.size(); i++)
	      {
		std::vector<double> met_data_line;
		met_data_line
		  .push_back(analytic_solution.get_analytic_air_temperature(all_dates_in_seconds[i][0]));
		met_data_line
		  .push_back(analytic_solution.get_analytic_relative_humidity(/*initial_time*/));
		met_data_line
		  .push_back(analytic_solution.get_analytic_wind_speed(/*initial_time*/));
		met_data_line
		  .push_back(/*wind direction*/0.);
		met_data_line
		  .push_back(analytic_solution.get_analytic_solar_radiation(all_dates_in_seconds[i][0]));
		met_data_line
		  .push_back(analytic_solution.get_analytic_precipitation(/*initial_time*/));
		// there are negative entries in solar radiation and positive entries in the 
		// middle of the night, this should help fix it.
		// if (met_data_line[4]<0.)
		//   met_data_line[4]=0;
		// if (((date_and_time[i][3]>=21)||
		//      (date_and_time[i][3]<=2))&&
		//     (met_data_line[4]>0.))
		//   met_data_line[4]=0;

		met_data.push_back(met_data_line);
	      }
	    //output met data
	    std::string output_filename="met_data_ph_";
	    std::ostringstream os;
	    os << preheating_step;
	    output_filename+=os.str() + ".txt";
	    std::ofstream file (output_filename.c_str());
	    if (!file.is_open())
	      throw 2;
	    pcout << "Writing file: " << output_filename.c_str() << std::endl;
	    data_tools.print_data(file,
				  met_data,
				  date_and_time);
	    file.close();
	    if (file.is_open())
	      throw 3;
	  }
	else
	  {
	    DataTools data_tools;
	    data_tools.read_met_data(date_and_time,
				     met_data,
				     time_step,
				     preheating_step,
				     met_data_type,
				     input_path);
	  }
	pcout << "\tAvailable met lines: " << met_data.size() 
	      << std::endl << std::endl;
      }
    old_air_temperature   = met_data[timestep_number-1][0];
    old_relative_humidity = met_data[timestep_number-1][1];
    old_wind_speed        = met_data[timestep_number-1][2];
    old_wind_direction    = met_data[timestep_number-1][3];
    old_solar_radiation   = met_data[timestep_number-1][4];
    old_precipitation     = met_data[timestep_number-1][5];
	
    new_air_temperature   = met_data[timestep_number][0];
    new_relative_humidity = met_data[timestep_number][1];
    new_wind_speed        = met_data[timestep_number][2];
    new_wind_direction    = met_data[timestep_number][3];
    new_solar_radiation   = met_data[timestep_number][4];
    new_precipitation     = met_data[timestep_number][5];
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  void Heat_Pipe<dim>::initial_condition_temperature()
  {
	  if (preheating==true)
	  {
		  std::ifstream file (preheating_input_filename.c_str());
		  if (!file.is_open())
			  throw 2;

		  Vector<double> initial_condition;
		  initial_condition.block_read (file);

		  backup_solution_temperature=initial_condition;

		  file.close();
		  if (file.is_open())
			  throw 3;
	  }
	  else
	  {
		  VectorTools::project(dof_handler_temperature,
				  constraints_temperature, QGauss<dim>(3),
				  ConstantFunction<dim>(10.),
				  backup_solution_temperature);
		  backup_solution_temperature.compress (VectorOperation::insert);
	  }
  }
  //**************************************************************************************
  //--------------------------------------------------------------------------------------
  //**************************************************************************************
  template<int dim>
  void Heat_Pipe<dim>::run()
  {
    TimerOutput timer (mpi_communicator,
    		       pcout,
    		       TimerOutput::summary,
    		       TimerOutput::cpu_and_wall_times);
    
    pcout << "\tOutput files (prefix): " 
    	  << preheating_output_filename << std::endl
    	  << std::endl;
    {
      TimerOutput::Scope timer_section(timer,"Read grid");
      read_mesh_temperature();
    }
    /*
      We need to initialize all vector and matrices and we need to 
      project the initial condition into the old_solution vector.
    */
    {
      TimerOutput::Scope timer_section(timer,"Setup system");
      setup_system_temperature();
    }
    {
      TimerOutput::Scope timer_section(timer,"Set initial condition");
      initial_condition_temperature();
      old_solution_temperature=backup_solution_temperature;
    }
    {
      TimerOutput::Scope timer_section(timer,"Mesh info");
      mesh_info();
      //      refine_grid();
      //      pcout << "\n\n";
      //      mesh_info();
    }
    /*
      It looks like the system was active whenever the temperature difference between
      the collector and storage was greater than 1.4 C and once was active it stop
      when the difference drop below 0.3 C.
      We define the control switch here so that it keeps its last value, and if it
      was active, and the temperature difference is greater than 0.3 it will remain
      active.
    */
    bool switch_control=false;
    bool previous_switch_control=false;

    double new_control_temperature_collector=0.;
    double new_control_temperature___storage=0.;

    double old_control_temperature_collector=10.;

    //double previous_new_collector_max_norm=0.;
    //double previous_new_storage___max_norm=0.;
    double previous_new_collector_avg_norm=0.;
    double previous_new_storage___avg_norm=0.;
    //double previous_new_collector_min_norm=0.;
    //double previous_new_storage___min_norm=0.;

    double current__new_collector_max_norm=0.;
    double current__new_storage___max_norm=0.;
    double current__new_collector_avg_norm=0.;
    double current__new_storage___avg_norm=0.;
    double current__new_collector_min_norm=0.;
    double current__new_storage___min_norm=0.;

    std::vector<double> old_inlet__temperatures_pipes(number_of_pipes,0);
    std::vector<double> old_outlet_temperatures_pipes(number_of_pipes,0);
    std::vector<double> previous_new_inlet__temperatures_pipes(number_of_pipes,0);
    std::vector<double> previous_new_outlet_temperatures_pipes(number_of_pipes,0);
    std::vector<double> current__new_inlet__temperatures_pipes(number_of_pipes,0);
    std::vector<double> current__new_outlet_temperatures_pipes(number_of_pipes,0);

    previous_new_soil_avg_surface_temperature=10.;
    previous_new_road_avg_surface_temperature=10.;
    previous_new_soil_max_surface_temperature=10.;
    previous_new_soil_min_surface_temperature=10.;
    previous_new_road_max_surface_temperature=10.;
    previous_new_road_min_surface_temperature=10.;

    old_avg_soil_surface_temperature=10.;
    current_new_avg_soil_surface_temperature=10.;
    old_avg_road_surface_temperature=10.;
    current_new_avg_road_surface_temperature=10.;

    current_new_max_soil_surface_temperature=0.;
    current_new_min_soil_surface_temperature=0.;
    current_new_max_road_surface_temperature=0.;
    current_new_min_road_surface_temperature=0.;
    old_max_soil_surface_temperature=0.;
    old_min_soil_surface_temperature=0.;
    old_max_road_surface_temperature=0.;
    old_min_road_surface_temperature=0.;
    /*
      The actual loop in time. The maximum time for the simulation, time
      step, initial time and many other variables are defined in the constructor
    */
    for (timestep_number=1, time=time_step;
    	 timestep_number<=timestep_number_max;
    	 timestep_number++, time+=time_step)
      { 
    	/*
     	 * Update meteorological data
     	 */
    	{
    		TimerOutput::Scope timer_section (timer,"Update met data");
    		update_met_data ();
    	}
    	/*
	  Prepare vectors with data at borehole's sensor depths.
	  We do this every time step.
    	 */
    	{
    		TimerOutput::Scope timer_section (timer,"Fill output vectors");
    		fill_output_vectors();
    	}
    	/*
	  Print the date, met data, and some micellaneous
	  info about the surface temperature
    	 */
    	/*
	  In the experimental data provided by TRL, in the sensor located nearest to the
	  surface, it can be observed a change in the daily variations of temperature for
	  the months of December to March. It is believe that this effect is produced by 
	  different canopy densities above the soil's surface. We are assuming that no 
	  more than 20 years are being simulated
    	 */
	canopy_density=0.85;
	if (author=="Best") // increase canopy density between october and february
	  if (date_and_time[timestep_number][1]>=10 &&
	      date_and_time[timestep_number][1]<=2)
	    canopy_density=0.85; //0.95
	/*
	  Here we define and initialize variables for convergence 
	  criteriums. First, those corresponding to the surface 
	  temperature convergence. We also need a
	  variable to tell us how many times have the loop been
	  performed in each time step.
	*/
	current__new_collector_max_norm=0.;
	current__new_storage___max_norm=0.;
	current__new_collector_avg_norm=0.;
	current__new_storage___avg_norm=0.;
	current__new_collector_min_norm=0.;
	current__new_storage___min_norm=0.;

//	double tolerance_collector_max_norm=-1000.;
//	double tolerance_storage___max_norm=-1000.;
	double tolerance_collector_avg_norm=-1000.;
	double tolerance_storage___avg_norm=-1000.;
//	double tolerance_collector_min_norm=-1000.;
//	double tolerance_storage___min_norm=-1000.;
	double tolerance_soil_avg_surface_temperature=-1000.;
	double tolerance_road_avg_surface_temperature=-1000.;
//	double tolerance_soil_max_surface_temperature=-1000.;
//	double tolerance_soil_min_surface_temperature=-1000.;
//	double tolerance_road_max_surface_temperature=-1000.;
//	double tolerance_road_min_surface_temperature=-1000.;

	double tolerance_limit_soil=0.4;
	double tolerance_limit_road=0.4;
	if (date_and_time[timestep_number][1]==6)
	  tolerance_limit_soil=0.8;

	unsigned int step=0;
	/*
	  Then those corresponding to the pipe system convergence. We to 
	  compare the heat flux at the collector and storage pipes in the
	  following way: average the collector heat flux for all pipes and
	  compare the new and old values. When the difference beween these
	  values is less than 10 Watts, break the loop.
	  We also need vectors that define the current state of the pipe
	  system. We will work with these temporal vectors and we will 
	  update the originals once the solution converges. These vectors
	  are meant to store just the current state of the system, not the
	  state through the whole simulation.
	  Here we also need a variable to tell us how many time has this
	  loop been perfomed.
	*/
	/*
	  We want to store the soil and road heat and mass fluxes at each
	  time step. This is done in a vectors of vectors, a kind of matrix.
	  Every line corresponds to a time step. As we don't know a priori
	  how many time steps are going to be performed, we add a new line
	  every time we enter a new time step. The line is rewritten 
	  everytime the system is assembled and when the system converges
	  it will store the final heat fluxes for that time step.
	*/
	{
	  std::vector <double> temp (number_of_surface_heat_and_mass_fluxes,0.);
	  road_heat_fluxes.push_back(temp);
	  soil_heat_fluxes.push_back(temp);
	  temp.clear();
	  temp.push_back(0.);
	  temp.push_back(0.);
	  control_temperatures.push_back(temp);
	}
	for (unsigned int i=0; i<number_of_pipes; i++)
	  current__new_pipe_heat_flux[i]=0.;
	
	do
	  { /*surface temperature convergence*/
	    for (unsigned int i=0; i<road_heat_fluxes[timestep_number-1].size(); i++)
	      road_heat_fluxes[timestep_number-1][i]=0.;
	    for (unsigned int i=0; i<soil_heat_fluxes[timestep_number-1].size(); i++)
	      soil_heat_fluxes[timestep_number-1][i]=0.;
	    {
	      TimerOutput::Scope timer_section (timer,"Assemble temperature");
	      assemble_system_parallel_temperature(/*step,*/switch_control);
	      assemble_system_petsc_temperature();
	    }
	    {
	      TimerOutput::Scope timer_section (timer,"Solve temperature");
	      solve_temperature();
	    }
	    double avg_collector__inlets=0.;
	    double avg_collector_outlets=0.;
	    double avg_storage____inlets=0.;
	    double avg_storage___outlets=0.;
 	    {
	      TimerOutput::Scope timer_section (timer,"Update surface temperatures");
	      previous_new_soil_avg_surface_temperature
		=current_new_avg_soil_surface_temperature;
	      previous_new_soil_max_surface_temperature
		=current_new_max_soil_surface_temperature;
	      previous_new_soil_min_surface_temperature
		=current_new_min_soil_surface_temperature;
	      previous_new_road_avg_surface_temperature
		=current_new_avg_road_surface_temperature;
	      previous_new_road_max_surface_temperature
		=current_new_max_road_surface_temperature;
	      previous_new_road_min_surface_temperature
		=current_new_min_road_surface_temperature;

	      // previous_new_collector_max_norm
	      // 	=current__new_collector_max_norm;
	      // previous_new_storage___max_norm
	      // 	=current__new_storage___max_norm;
	      previous_new_collector_avg_norm
		=current__new_collector_avg_norm;
	      previous_new_storage___avg_norm
		=current__new_storage___avg_norm;
	      // previous_new_collector_min_norm
	      // 	=current__new_collector_min_norm;
	      // previous_new_storage___min_norm
	      // 	=current__new_storage___min_norm;
		      
 	      surface_temperatures();

	    current__new_collector_max_norm=0.;
	    current__new_collector_avg_norm=0.;
	    current__new_collector_min_norm=0.;
	    current__new_storage___max_norm=0.;
	    current__new_storage___avg_norm=0.;
	    current__new_storage___min_norm=0.;
	    for (unsigned int i=0; i<number_of_pipes/2; i++)
	      {
		current__new_collector_max_norm
		  +=max_pipe_temperature[i]/(number_of_pipes/2);
		current__new_collector_avg_norm
		  +=new_avg_pipe_temperature[i]/(number_of_pipes/2);
		current__new_collector_min_norm
		  +=min_pipe_temperature[i]/(number_of_pipes/2);

		current__new_storage___max_norm
		  +=max_pipe_temperature[(number_of_pipes/2)+i]/(number_of_pipes/2);
		current__new_storage___avg_norm
		  +=new_avg_pipe_temperature[(number_of_pipes/2)+i]/(number_of_pipes/2);
		current__new_storage___min_norm
		  +=min_pipe_temperature[(number_of_pipes/2)+i]/(number_of_pipes/2);

		if (i<number_of_pipes/4)
		  {
		    avg_collector__inlets
		      +=max_pipe_temperature[i]/(number_of_pipes/4);
		    avg_storage____inlets
		      +=max_pipe_temperature[(number_of_pipes/2)+i]/(number_of_pipes/4);
		    avg_collector_outlets
		      +=max_pipe_temperature[(number_of_pipes/4)+i]/(number_of_pipes/4);
		    avg_storage___outlets
		      +=max_pipe_temperature[(number_of_pipes/4)+(number_of_pipes/2)+i]/(number_of_pipes/4);
		  }
 	      }
	      /*define tolerances*/
	    tolerance_soil_avg_surface_temperature
		=fabs(current_new_avg_soil_surface_temperature
				-previous_new_soil_avg_surface_temperature);
	    tolerance_road_avg_surface_temperature
		=fabs(current_new_avg_road_surface_temperature
				-previous_new_road_avg_surface_temperature);
//	    tolerance_soil_max_surface_temperature
//		=fabs(current_new_max_soil_surface_temperature
//				-previous_new_soil_max_surface_temperature);
//	    tolerance_soil_min_surface_temperature
//		=fabs(current_new_min_soil_surface_temperature
//				-previous_new_soil_min_surface_temperature);
//	    tolerance_road_max_surface_temperature
//		=fabs(current_new_max_road_surface_temperature
//				-previous_new_road_max_surface_temperature);
//	    tolerance_road_min_surface_temperature
//		=fabs(current_new_min_road_surface_temperature
//				-previous_new_road_min_surface_temperature);

	    // tolerance_collector_max_norm
	    // 	=fabs(current__new_collector_max_norm
	    // 			-previous_new_collector_max_norm);
	    // tolerance_storage___max_norm
	    // 	=fabs(current__new_storage___max_norm
	    // 			-previous_new_storage___max_norm);
	    tolerance_collector_avg_norm
	    	=fabs(current__new_collector_avg_norm
	    			-previous_new_collector_avg_norm);
	    tolerance_storage___avg_norm
	    	=fabs(current__new_storage___avg_norm
	    			-previous_new_storage___avg_norm);
	    // tolerance_collector_min_norm
	    // 	=fabs(current__new_collector_min_norm
	    // 			-previous_new_collector_min_norm);
	    // tolerance_storage___min_norm
	    // 	=fabs(current__new_storage___min_norm
	    // 			-previous_new_storage___min_norm);
 	    }
	    //==========================================================================================
	    // /*-----solve pipe system ------*/
	    if (step==0)
	      {
		if (preheating_step==4 ||
		    preheating_step==7)
		  {
		    new_control_temperature_collector
		      =current__new_collector_avg_norm;
		    new_control_temperature___storage
		      =current__new_storage___avg_norm;
		  }
		else if (preheating_step==5 ||
			 preheating_step==8)
		  {
		    const Vector<double> localized_solution_temperature(old_solution_temperature);

		    new_control_temperature_collector
		      =VectorTools::point_value(dof_handler_temperature,
						localized_solution_temperature,
						Point<dim>(0.,-0.025));
		    new_control_temperature___storage
		      =current__new_storage___avg_norm;
		  }
		control_temperatures[timestep_number-1][0]
		  =new_control_temperature_collector;
		control_temperatures[timestep_number-1][1]
		  =new_control_temperature___storage;
		//}
		// previous_new_pipe_heat_flux
		//   =current__new_pipe_heat_flux;
		/*
		  Here is implemented the activation of the system. There are two main ways:
	      
		  --1-- The system is activated automatically if a certain criteria is meet.
		  At the moment this criteria is the temperature difference between the
		  average temperature at the surface of collector and storage pipes. Two 
		  main periods of time are identified in the experimental results:
		  from 23/08/2005 to 14/11/2005 and from 15/11/2005 to 20/02/2006
	      
		  --2-- The system is forced to be active in certain ranges of time (i.e.
		  between noon and 10pm everyday).
		*/
		if (activation_type=="automatic_activation")
		  {
		    // first period: from 23/08/2005 to 14/11/2005
		    if (preheating_step==4 ||
			preheating_step==7)
		      {
			if (((new_control_temperature_collector-new_control_temperature___storage)>1.4) || //1.4
			    (switch_control && 
			     ((new_control_temperature_collector-new_control_temperature___storage)>0.4))) //0.4
			  switch_control=true;
			else
			  switch_control=false;
			/*
			  If there is any interval where the system is off or on for any external
			  reason (e.g. failure) modify the followig IF statement.
			*/
			if ((preheating_step==4)&&
			    (date_and_time[timestep_number][0]==23 && 
			     date_and_time[timestep_number][1]==8  && 
			     date_and_time[timestep_number][2]==2005 && 
			     (date_and_time[timestep_number][3]<13 || 
			      (date_and_time[timestep_number][3]==13 && 
			       date_and_time[timestep_number][4]<0))))
			  switch_control=false;
			else if ((preheating_step==7)&&
				 ((date_and_time[timestep_number][1]==5  && 
				   date_and_time[timestep_number][2]==2006 && 
				   date_and_time[timestep_number][0]>=18)||
				  (date_and_time[timestep_number][1]==6  && 
				   date_and_time[timestep_number][2]==2006 && 
				   date_and_time[timestep_number][0]<5)))
			  switch_control=false;
		      }
		    else if (preheating_step==5 ||// second period: from 15/11/2005 to 20/02/2006
			     preheating_step==8)
		      {
			double control_temperature=4.;
			if ((old_control_temperature_collector<control_temperature)&&
			    (new_control_temperature_collector<control_temperature))
			  switch_control=true;
			else if (switch_control &&
				 (old_control_temperature_collector>control_temperature)&&
				 (new_control_temperature_collector>control_temperature))
			  switch_control=false;
			else 
			  switch_control=false;
		
			if (new_control_temperature___storage<
			    new_control_temperature_collector)
			  switch_control=false;
		      }
		    else
		      switch_control=false;
		  }
		else if (activation_type=="forced_activation")
		  {
		    if ((date_and_time[timestep_number][3]>=14) &&
			(date_and_time[timestep_number][3]<=20) &&
			(date_and_time[timestep_number][1]<10))
		      switch_control=true;
		    else
		      switch_control=false;
		  }
		else
		  {
		    pcout << "Error. Wrong activation type\n";
		    throw -1;
		  }
 	      }

	    if ((pipe_system==true)&&
		(switch_control==true))
	      {
		if ((previous_switch_control==false)&&
		    (switch_control==true)&&
		    (step==0))
		  {
		    for (unsigned int i=0; i<number_of_pipes/2; i++)
		      {
			current__new_inlet__temperatures_pipes[   i]
			  =new_avg_pipe_temperature[20+i];
			current__new_inlet__temperatures_pipes[20+i]
			  =new_avg_pipe_temperature[   i];
			
			old_pipe_heat_flux[   i]=0.;
			old_pipe_heat_flux[20+i]=0.;
		      }
		  }
		else
		  current__new_inlet__temperatures_pipes
		    =previous_new_inlet__temperatures_pipes;
		
		std::vector<double> current_heat_flux_temp(number_of_pipes,0);
		PipeSystem system_pipes(time_step);

		system_pipes.pipe_heat_flux(new_avg_pipe_temperature,
					    old_avg_pipe_temperature,
					    current__new_inlet__temperatures_pipes,
					    current__new_outlet_temperatures_pipes,
					    current_heat_flux_temp);
		for (unsigned int i=0; i<number_of_pipes; i++)
		  current__new_pipe_heat_flux[i]
		    +=current_heat_flux_temp[i];
		
		previous_new_inlet__temperatures_pipes
		  =current__new_inlet__temperatures_pipes;
	      }
	    step++;
	    cell_index_to_previous_new_surface_temperature
	      =cell_index_to_current__new_surface_temperature;
	  }while ((tolerance_soil_avg_surface_temperature>tolerance_limit_soil) ||
		  (tolerance_road_avg_surface_temperature>tolerance_limit_road) ||
		  ((tolerance_collector_avg_norm>0.1)&&(pipe_system==true)&&(switch_control==true))||
		  ((tolerance_storage___avg_norm>0.1)&&(pipe_system==true)&&(switch_control==true)));

	if(pcout.is_active())
	  {
	    std::cout << "Time step " << timestep_number << "\t"
		      << std::setw(2) << std::setfill('0') << date_and_time[timestep_number][0] << "/" 
		      << std::setw(2) << std::setfill('0') << date_and_time[timestep_number][1] << "/" 
		      << std::setw(2) << std::setfill('0') << date_and_time[timestep_number][2] << "\t" 
		      << std::setw(2) << std::setfill('0') << date_and_time[timestep_number][3] << ":" 
		      << std::setw(2) << std::setfill('0') << date_and_time[timestep_number][4] << ":" 
		      << std::setw(2) << std::setfill('0') << date_and_time[timestep_number][5];
	    std::cout.setf( std::ios::fixed, std::ios::floatfield );
	    std::cout << "\tTa" << std::setw(7) << std::setfill(' ') << std::setprecision(2) << new_air_temperature 
		      << "\tRs" << std::setw(7) << std::setfill(' ') << std::setprecision(2) << new_solar_radiation 
		      << "\tUs" << std::setw(7) << std::setfill(' ') << std::setprecision(2) << new_wind_speed 
		      << "\tHr" << std::setw(7) << std::setfill(' ') << std::setprecision(2) << new_relative_humidity 
		      << "\tI " << std::setw(7) << std::setfill(' ') << std::setprecision(2) << new_precipitation
		      << "\tc " << std::setw(7) << std::setfill(' ') << std::setprecision(2) << canopy_density
	      	      << "\ts " << std::setw(7) << std::setfill(' ') << std::setprecision(2) << step
		      << std::endl;
	  }
 	{
	  /*
	    Output the solution at the beggining, end and every
	    certain time steps in this case every:
	  */
	  if ((preheating_step>3) && timestep_number%12==0 /*&&
	      ((timestep_number==2) ||
	      (timestep_number==timestep_number_max))*/ /* ||
	       ((date_and_time[timestep_number][3]%6==0)&&
	       (date_and_time[timestep_number][4]==30))))*/
	      // ((date_and_time[timestep_number][0]==9)&&
	      //  (date_and_time[timestep_number][1]==9)&&
	      //  (date_and_time[timestep_number][2]==2005)&&
	      //  (date_and_time[timestep_number][3]==12)&&
	      //  (date_and_time[timestep_number][4]==30))||
	      // ((date_and_time[timestep_number][0]==9)&&
	      //  (date_and_time[timestep_number][1]==9)&&
	      //  (date_and_time[timestep_number][2]==2005)&&
	      //  (date_and_time[timestep_number][3]==20)&&
	      //  (date_and_time[timestep_number][4]==30))||
	      // ((date_and_time[timestep_number][0]==10)&&
	      //  (date_and_time[timestep_number][1]==9)&&
	      //  (date_and_time[timestep_number][2]==2005)&&
	      //  (date_and_time[timestep_number][3]==3)&&
	      //  (date_and_time[timestep_number][4]==30))||
	      // ((date_and_time[timestep_number][0]==27)&&
	      //  (date_and_time[timestep_number][1]==9)&&
	      //  (date_and_time[timestep_number][2]==2005)&&
	      //  (date_and_time[timestep_number][3]==12)&&
	      //  (date_and_time[timestep_number][4]==30))||
	      // ((date_and_time[timestep_number][0]==27)&&
	      //  (date_and_time[timestep_number][1]==9)&&
	      //  (date_and_time[timestep_number][2]==2005)&&
	      //  (date_and_time[timestep_number][3]==20)&&
	      //  (date_and_time[timestep_number][4]==30))||
	      // ((date_and_time[timestep_number][0]==28)&&
	      //  (date_and_time[timestep_number][1]==9)&&
	      //  (date_and_time[timestep_number][2]==2005)&&
	      //  (date_and_time[timestep_number][3]==3)&&
	      //  (date_and_time[timestep_number][4]==30))||
	      // ((date_and_time[timestep_number][0]==30)&&
	      //  (date_and_time[timestep_number][1]==10)&&
	      //  (date_and_time[timestep_number][2]==2005)&&
	      //  (date_and_time[timestep_number][3]==0)&&
	      //  (date_and_time[timestep_number][4]==30))||
	      // ((date_and_time[timestep_number][0]==2)&&
	      //  (date_and_time[timestep_number][1]==11)&&
	      //  (date_and_time[timestep_number][2]==2005)&&
	      //  (date_and_time[timestep_number][3]==0)&&
	      //  (date_and_time[timestep_number][4]==30))||
	      // ((date_and_time[timestep_number][0]==4)&&
	      //  (date_and_time[timestep_number][1]==11)&&
	      //  (date_and_time[timestep_number][2]==2005)&&
	      //  (date_and_time[timestep_number][3]==0)&&
	      //  (date_and_time[timestep_number][4]==30))||
	      // ((date_and_time[timestep_number][0]==6)&&
	      //  (date_and_time[timestep_number][1]==11)&&
	      //  (date_and_time[timestep_number][2]==2005)&&
	      //  (date_and_time[timestep_number][3]==0)&&
	      //  (date_and_time[timestep_number][4]==30))||
	      // ((date_and_time[timestep_number][0]==8)&&
	      //  (date_and_time[timestep_number][1]==11)&&
	      //  (date_and_time[timestep_number][2]==2005)&&
	      //  (date_and_time[timestep_number][3]==0)&&
	      //  (date_and_time[timestep_number][4]==30))
	      )
	    {
	      TimerOutput::Scope timer_section (timer,"Output results");
	      output_results();
	    }
	  /*
	    At the end of the simulation we output the previous
	    defined vectors and visualization files
	  */
	  if (timestep_number==timestep_number_max)
	    {
	      TimerOutput::Scope timer_section (timer,"Output text files");
	      const Vector<double> localized_solution_temperature (solution_temperature);
	      if (this_mpi_process==0)
		{
		  std::vector< std::vector<int> >::const_iterator 
		    first=date_and_time.begin(), second=date_and_time.begin()+timestep_number_max;
		  std::vector< std::vector<int> > date_and_time_1d(first,second);
		      
		  // Print file with data at borehole sensor's depths
		  // generate suffix
		  {
		    std::vector< std::string > filenames;
		    filenames.push_back(preheating_output_filename+"_bha_temperature.txt");
		    filenames.push_back(preheating_output_filename+"_bhf_temperature.txt");
 		    filenames.push_back(preheating_output_filename+"_bhh_temperature.txt");
		    filenames.push_back(preheating_output_filename+"_bhi_temperature.txt");
		    filenames.push_back(preheating_output_filename+"_road_heat_fluxes.txt");
		    filenames.push_back(preheating_output_filename+"_soil_heat_fluxes.txt");
		    if (pipe_system==true)
		      {
			filenames.push_back(preheating_output_filename+"_pipe_heat_fluxes.txt");
			filenames.push_back(preheating_output_filename+"_control_temperatures.txt");
		      }
		    std::vector< std::vector< std::vector<double> > > data;
		    data.push_back(soil_bha_temperature);
		    data.push_back(soil_bhf_temperature);
		    data.push_back(soil_bhh_temperature);
		    data.push_back(soil_bhi_temperature);
		    data.push_back(road_heat_fluxes);
		    data.push_back(soil_heat_fluxes);
		    if (pipe_system==true)
		      {
			data.push_back(pipe_heat_fluxes);
			data.push_back(control_temperatures);
		      }
		    
		    for (unsigned int f=0; f<filenames.size(); f++)
		      {
			std::ofstream file (filenames[f].c_str());
			if (!file.is_open())
			  throw 2;
			pcout << "Writing file: " << filenames[f].c_str() << std::endl;
			DataTools data_tools;
			data_tools.print_data(file,
					      data[f],
					      date_and_time_1d);
			file.close();
			if (file.is_open())
			  throw 3;
		      }
 		  }
		  // Print preheating files
		  {
		    std::ofstream file_temperature (preheating_output_filename.c_str());			
		    if (!file_temperature.is_open())
		      throw 2;		      
		    localized_solution_temperature.block_write(file_temperature);
		    file_temperature.close();
		    if (file_temperature.is_open())
		      throw 3;
		  }
 		}
 	    }
 	}
	/*
	  update old solution to the current solution
	  update everything that need to be updated
	*/
 	{
 	  old_solution_temperature=solution_temperature;

	  cell_index_to_old_surface_temperature
	    =cell_index_to_current__new_surface_temperature;
	  old_avg_soil_surface_temperature
	    =current_new_avg_soil_surface_temperature;
	  old_avg_road_surface_temperature
	    =current_new_avg_road_surface_temperature;

	  old_max_soil_surface_temperature
	    =current_new_max_soil_surface_temperature;
	  old_min_soil_surface_temperature
	    =current_new_min_soil_surface_temperature;
	  old_max_road_surface_temperature
	    =current_new_max_road_surface_temperature;
	  old_min_road_surface_temperature
	    =current_new_min_road_surface_temperature;

	  old_avg_pipe_temperature
	    =new_avg_pipe_temperature;
	   
	  old_inlet__temperatures_pipes
	    =current__new_inlet__temperatures_pipes;
	  old_outlet_temperatures_pipes
	    =current__new_outlet_temperatures_pipes;

	  cell_index_3d_to_old_pipe_heat_flux
	    =cell_index_3d_to_current__new_pipe_heat_flux;
	  old_pipe_heat_flux
	    =current__new_pipe_heat_flux;

//	  old_collector_avg_norm
//	    =current__new_collector_avg_norm;
//	  old_storage___avg_norm
//	    =current__new_storage___avg_norm;

	  previous_switch_control=switch_control;
	  old_control_temperature_collector
	    =new_control_temperature_collector;
 	}
      }
    /*
      Nothing better than a job well done (hopefully)
    */
    pcout << std::endl
    	  << std::endl << "\t"
    	  << "\t Job Done!!"
    	  << std::endl;
  }
}
//**************************************************************************************
//--------------------------------------------------------------------------------------
//**************************************************************************************
int main(int argc, char **argv)
{
  try
    {  
      using namespace TRL;
      using namespace dealii;
      const unsigned int dim=2;
      
      //PetscInitialize(&argc,&argv,0,0);
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv,1);
      // for (unsigned int i=0; i<argc; i++)
      // 	std::cout << argv[i] << "\t";
      // std::cout << "\n";

      {
      	deallog.depth_console (0);
	
      	FE_Q<dim> fe(1);
      	Heat_Pipe<dim> trl_problem(argc,argv,fe);
      	trl_problem.run ();
      }
      //PetscFinalize();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Exception on processing: " << std::endl
		<< exc.what() << std::endl
		<< "Aborting!" << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }   
  
   return 0;
}
