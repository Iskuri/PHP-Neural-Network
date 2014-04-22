<?php

/*
*
* XOR non linear training test
*/

include("NeuralNetwork.php");

$network = new NeuralNetwork(array(2,3,3,1));

for($k = 0 ; $k < 100000 ; $k++) {

	$outVals = array();

	for($i = 0 ; $i < 2 ; $i++) {

		for($j = 0 ; $j < 2 ; $j++) {

			$outVal = $i ^ $j;

			$network->setInput(0,$i);
			$network->setInput(1,$j);

			$network->trainNetwork(0,$outVal);

			$outVals[] = $outVal - $network->getOutput(0);
		}
	}

	echo implode(" : ",$outVals)."\n";
}


?>
