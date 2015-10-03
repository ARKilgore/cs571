package edu.emory.mathcs.nlp.component.dep;

import edu.emory.mathcs.nlp.learn.util.Prediction;

public class DEPStatePrediction<S> {

	public S state;
	public Prediction prediction;
	
	public DEPStatePrediction(S s, Prediction p) {
		this.state = s;
		this.prediction = p;
	}
	
	public S getState() {
		return state;
	}
	
	public void setState(S state) {
		this.state = state;
	}
	
	public Prediction getPrediction() {
		return prediction;
	}
	
	public void setPrediction(Prediction prediction) {
		this.prediction = prediction;
	}
	
}
