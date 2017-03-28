#!/bin/bash

#export LD_LIBRARY_PATH=$HOME/petsc-3.4.2/x86_64/lib:$LD_LIBRARY_PATH

CODE=mycode
TEST_TYPE=$1

SHADING_BOOL=true
SHADING_PX=ws
SHADING_FACTOR=0.40
BOTTOM_BC=fr
BOTTOM_BC_BOOL=false

MAINDIR=$HOME/git/ihts_2d/
INPUTPATH=$HOME/input
MESH=trl_mesh_in_2d.msh

######################################
# Define and create output directory #
######################################
OUTPUTPATH=$MAINDIR/output
# if [ "$TEST_TYPE" == "author" ]
# then
#     #    WDPATH=/scratch/$USER/third_paper_contour_plots #test_author
# elif [ "$TEST_TYPE" == "weather" ]
# then
#     #    WDPATH=/scratch/$USER/test_weather
# elif [ "$TEST_TYPE" == "thermal_conductivity" ]
# then
#     #    WDPATH=/scratch/$USER/test_thermal_conductivity    
# else
#     echo "Wrong test type. Current test type: "
#     echo $TEST_TYPE
#     exit 1
# fi
rm -rf $OUTPUTPATH
mkdir $OUTPUTPATH
######################################

######################################
# Define flags for program           #
######################################
for VARIATION in 0 #1 2 #0 1 2
do
    AUTHOR=Best
    THERMAL_CONDUCTIVITY_FACTOR=1.0
    TYPE_OF_WEATHER=emd-trl #amd-mild-1 #

    if [ "$TEST_TYPE" == "author" ]
    then
	if [ "$VARIATION" -eq 0 ]
	then
    	    AUTHOR="Best"
	elif [ "$VARIATION" -eq 1 ]
	then
    	    AUTHOR="Herb"
	elif [ "$VARIATION" -eq 2 ]
	then
    	    AUTHOR="Fixed"
	else
    	    echo "Wrong author option. Current option: "
    	    echo $AUTHOR
	    exit 1
	fi
    elif [ "$TEST_TYPE" == "weather" ]
    then
	if [ "$VARIATION" -eq 0 ]
	then
    	    TYPE_OF_WEATHER=amd-mild-2
	elif [ "$VARIATION" -eq 1 ]
	then
    	    TYPE_OF_WEATHER=amd-cold
	elif [ "$VARIATION" -eq 2 ]
	then
    	    TYPE_OF_WEATHER=amd-hot
	else
    	    echo "Wrong weather option. Current option: "
    	    echo $TYPE_OF_WEATHER
	    exit 1
	fi
    elif [ "$TEST_TYPE" == "thermal_conductivity" ]
    then
	if [ "$VARIATION" -eq 0 ]
	then
    	    THERMAL_CONDUCTIVITY_FACTOR=0.8
	elif [ "$VARIATION" -eq 1 ]
	then
    	    THERMAL_CONDUCTIVITY_FACTOR=1.0
	elif [ "$VARIATION" -eq 2 ]
	then
    	    THERMAL_CONDUCTIVITY_FACTOR=1.2
	else
    	    echo "Thermal conductivity factor out of range. Current value: "
	    echo $THERMAL_CONDUCTIVITY_FACTOR
	    exit 1
	fi
    else
	echo "Wrong test type. Current test type: "
	echo $TEST_TYPE
	exit 1
    fi
    
    for VAR in 0 #0 1 2 3 4 5 6 7 # 
    do
	if [ "$VAR" -eq 0 ]
	then
    	    PreHS=1
    	    INITIAL_CONDITION=0
    	    INSULATION=false
    	    PIPE_SYSTEM=false
	elif [ "$VAR" -eq 1 ]
	then
    	    PreHS=2
    	    INITIAL_CONDITION=1
    	    INSULATION=false
    	    PIPE_SYSTEM=false
	elif [ "$VAR" -eq 2 ]
	then
    	    PreHS=3
    	    INITIAL_CONDITION=2
    	    INSULATION=true
    	    PIPE_SYSTEM=false
	elif [ "$VAR" -eq 3 ]
	then 
	    PreHS=4
	    INITIAL_CONDITION=3
	    INSULATION=true
	    PIPE_SYSTEM=true
	elif [ "$VAR" -eq 4 ]
	then
    	    PreHS=5
    	    INITIAL_CONDITION=4
    	    INSULATION=true
    	    PIPE_SYSTEM=true
	elif [ "$VAR" -eq 5 ]
	then
    	    PreHS=6
    	    INITIAL_CONDITION=5
    	    INSULATION=true
    	    PIPE_SYSTEM=false
	elif [ "$VAR" -eq 6 ]
	then
    	    PreHS=7
    	    INITIAL_CONDITION=6
    	    INSULATION=true
    	    PIPE_SYSTEM=true
	elif [ "$VAR" -eq 7 ]
	then
    	    PreHS=8
    	    INITIAL_CONDITION=7
    	    INSULATION=true
    	    PIPE_SYSTEM=true
	else
	    echo "Preheating step number out of range."
	    exit 1
	fi

	PREFIX=\
${AUTHOR}_ph\
${PreHS}_2d_\
${SHADING_PX}_\
${BOTTOM_BC}_\
${TYPE_OF_WEATHER}_\
${SHADING_FACTOR}_\
${THERMAL_CONDUCTIVITY_FACTOR}\
	
	echo ${PREFIX}
	ARGS=$(echo \
		   ${AUTHOR} \
		   ${PreHS} \
		   ${SHADING_BOOL} \
		   ${BOTTOM_BC_BOOL} \
		   ${TYPE_OF_WEATHER} \
		   ${INSULATION} \
		   ${PIPE_SYSTEM} \
		   ${SHADING_FACTOR} \
		   ${THERMAL_CONDUCTIVITY_FACTOR} \
		   ${INPUTPATH})
	echo ${ARGS}
	######################################
	
	mpirun -np 3 ./$CODE ${ARGS} > ${PREFIX}_output.txt

	rm -rf $OUTPUTPATH/${PREFIX}
	mkdir $OUTPUTPATH/${PREFIX}
	cp ${PREFIX}* $OUTPUTPATH/${PREFIX}
	cp ${PREFIX} $MAINDIR/preheatings/
	# cp *.txt $OUTPUTPATH/${PREFIX}
	cp $OUTPUTPATH/*.vtu $OUTPUTPATH/${PREFIX}
    done
done
