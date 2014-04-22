<?php

/**
 * Description of NeuralNetwork
 *
 * @author christopher.wade
 */
class NeuralNetwork {

	private $indices;

	private $allNeurons;

	protected $debugMode = false;

	public function __construct($_indices) {
		$this->indices = $_indices;

		$this->allNeurons = array();

		for ($i = 0; $i < count($_indices); $i++) {
			$this->allNeurons[$i] = array();
		}

		for ($i = 0; $i < count($this->allNeurons); $i++) {
			for ($j = 0; $j < $_indices[$i]; $j++) {

				$null = null;

				if ($i == 0) {
					$this->allNeurons[$i][$j] = new Neuron(1,$this->allNeurons[$i+1],$null);
				} elseif ($i == (count($this->allNeurons)-1)) {
					$this->allNeurons[$i][$j] = new Neuron($_indices[$i-1],$null,$this->allNeurons[$i-1]);
				} else {
					$this->allNeurons[$i][$j] = new Neuron($_indices[$i-1],$this->allNeurons[$i+1],$this->allNeurons[$i-1]);
				}
			}
		}

		//die;
		if ($this->debugMode) echo "Layer count for NN: ".(count($this->allNeurons))."\n";

	}

	public function getOutput($index) {
		if ($this->debugMode) echo "Layer count for NN: ".(count($this->allNeurons))."\n";
		if ($this->debugMode) echo "Getting output for index {$index}\n";

		for ($i = 0; $i < count($this->allNeurons); $i++) {
			if ($this->debugMode) echo "Calculating for layer: {$i}\n";

			for ($j = 0; $j < count($this->allNeurons[$i]); $j++) {
				if ($this->debugMode) echo "Calculating for neuron: {$j} on layer {$i}\n";

				$this->allNeurons[$i][$j]->pushOutputs($j);
			}
		}

		return $this->allNeurons[count($this->allNeurons)-1][$index]->output;
	}

	public function flushErrorSignal() {
		for ($i = 0; $i < count($this->allNeurons); $i++) {
			for ($j = 0; $j < count($this->allNeurons[$i]); $j++) {
				$this->allNeurons[$i][$j]->errorSignal = 0;
			}
		}
		if ($this->debugMode) echo "Layer count for NN: ".(count($this->allNeurons))."\n";
	}

	public function trainNetwork($index, $value) {
		if ($this->debugMode) echo "Flushing errors\n";

		$this->flushErrorSignal();

		if ($this->debugMode) echo "Getting outputs\n";

		$this->getOutput($index);

		if ($this->debugMode) echo "Calculating error\n";

		$this->allNeurons[count($this->allNeurons)-1][$index]->errorSignal = $value - $this->allNeurons[count($this->allNeurons)-1][$index]->output;

		if ($this->debugMode) echo "Layer count for NN: ".(count($this->allNeurons))."\n";

		for ($i = (count($this->allNeurons)-1); $i >= 0; $i--) {
			if ($this->allNeurons[$i] == $this->allNeurons[(count($this->allNeurons)-1)]) {
				if ($this->debugMode) echo "Pushing Initial Error, {$i}, {$index}\n";
				$this->allNeurons[$i][$index]->pushError();
			} else {
				foreach ($this->allNeurons[$i] as $key=> $tempNeuron) {
					if ($this->debugMode) echo "Pushing Neuron Error: {$i}, {$key}\n";
					$this->allNeurons[$i][$key]->pushError();
				}
			}
		}

		for ($i = 0; $i < count($this->allNeurons); $i++) {
			for ($j = 0; $j < count($this->allNeurons[$i]); $j++) {
				$this->allNeurons[$i][$j]->updateWeights();
			}
		}
	}

	public function setInput($index, $input) {
		$this->allNeurons[0][$index]->inputValues[0] = $input;
	}


}

class Neuron {

	public $inputValues;
	public $inputWeights;

	public $inputNeurons;
	public $outputNeurons;

	public $output;

	public $errorSignal;

	protected $learningCoefficient = 0.25;

	private $debugMode = false;

	public function __construct($valCount, &$_outNeurons, &$_inNeurons) {

		$this->inputNeurons = &$_inNeurons;
		$this->outputNeurons = &$_outNeurons;

		$this->inputValues = array();
		$this->inputWeights = array();

		$this->randomizeWeights($valCount);
	}

	public function randomizeWeights($total) {
		for ($i = 0; $i < $total; $i++) {
			$this->inputWeights[$i] = (rand()/getrandmax())*10 - 5;
			//echo "Random input weights: {$this->inputWeights[$i]}\n";
		}
	}

	public function pushOutputs($index) {
		$this->calcOutput();

		//echo "Output: {$this->output}\n";

		if ($this->outputNeurons) {
			for ($i = 0; $i < count($this->outputNeurons); $i++) {
				$this->outputNeurons[$i]->inputValues[$index] = $this->output;
				if ($this->debugMode) echo("Pushing output: {$this->output}\n");
			}
		} else {
			if ($this->debugMode) echo "WTF\n";
		}
	}

	public function updateWeights() {
		$derivative = $this->calcDerivative();

		if (!$this->inputNeurons) {
			for ($i = 0; $i < count($this->inputWeights); $i++) {
				$this->inputWeights[$i] = $this->inputWeights[$i] + ($this->learningCoefficient * $this->errorSignal * $derivative * $this->inputValues[$i]);
			}
		} else {
			for ($i = 0; $i < count($this->inputWeights); $i++) {

				$this->inputWeights[$i] = $this->inputWeights[$i] + ($this->learningCoefficient * $this->errorSignal * $derivative * $this->inputNeurons[$i]->output);
			}
		}

	}

	public function calcDerivative() {
		$increment = 0;

		for ($i = 0; $i < count($this->inputValues); $i++) {
			$increment += $this->inputValues[$i] * $this->inputWeights[$i];
		}

		$increment = $this->derivativeFunction($increment);

		return $increment;
	}

	public function calcOutput() {
		$increment = 0;

		for ($i = 0; $i < count($this->inputValues); $i++) {
			$increment += $this->inputValues[$i] * $this->inputWeights[$i];
		}

		$increment = $this->activationFunction($increment);

		$this->output = $increment;

		return $increment;

	}

	public function activationFunction($input) {
		return 1/(1+exp(-$input));
	}

	public function derivativeFunction($input) {
		$activation = $this->activationFunction($input);

		return $activation * (1-$activation);
	}

	public function pushError() {
		if ($this->inputNeurons) {
			for ($i = 0; $i < count($this->inputNeurons); $i++) {
				$this->inputNeurons[$i]->errorSignal += $this->errorSignal * $this->inputWeights[$i];
			}
		}
	}

}
