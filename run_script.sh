#!/bin/bash

function list_include_item {
    local list="$1"
    local item="$2"
    if [[ $list =~ (^|[[:space:]])"$item"($|[[:space:]]) ]] ; then
	#yes, list include the item
	result=0
    else
	result=1
    fi
    return $result
}

for ((ph=1; ph<2; ph++))
do
    with_pipe_system=false;
    with_insulation=false;
    
    if `list_include_item "4 5 7 8" "$ph"` ; then
	with_pipe_system=true
    else
	with_pipe_system=false
    fi

    if `list_include_item "3 4 5 6 7 8" "$ph"` ; then
	with_insulation=true
    else
	with_insulation=false
    fi

    echo -e $ph "\t" $with_pipe_system "\t" $with_insulation

    sed -i "s/\tset preheating step    =.*#/\tset preheating step    = $ph #/g" input.prm
    sed -i "s/\tset with insulation    =.*#/\tset with insulation    = $with_insulation #/g" input.prm
    sed -i "s/\tset with pipe system   =.*#/\tset with pipe system   = $with_pipe_system #/g" input.prm

    #cat input.prm > output_$ph.txt
    
    CODE=mycode
    
    mpirun -np 1 $CODE input.prm > output_$ph.txt
    
    mv output/*.vtu visualization
    mv output/*.txt summary_output
    mv output/* preheatings
done



