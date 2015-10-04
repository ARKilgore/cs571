package edu.emory.mathcs.nlp.component.dep.feature;

import edu.emory.mathcs.nlp.component.dep.DEPFeatureTemplate;
import edu.emory.mathcs.nlp.component.util.feature.Direction;
import edu.emory.mathcs.nlp.component.util.feature.FeatureItem;
import edu.emory.mathcs.nlp.component.util.feature.Field;
import edu.emory.mathcs.nlp.component.util.feature.Relation;
import edu.emory.mathcs.nlp.component.util.feature.Source;

public class DEPFeatureTemplate1 extends DEPFeatureTemplate{
	private static final long serialVersionUID = 4717085054409332081L;

	@Override
	protected void init()
	{
		// lemma features 
		add(new FeatureItem<>(Source.i, -1, Field.lemma));
		add(new FeatureItem<>(Source.i,  0, Field.lemma));
		add(new FeatureItem<>(Source.i,  1, Field.lemma));
		
		add(new FeatureItem<>(Source.j, -2, Field.lemma));
		add(new FeatureItem<>(Source.j, -1, Field.lemma));
		add(new FeatureItem<>(Source.j,  0, Field.lemma));
		add(new FeatureItem<>(Source.j,  1, Field.lemma));
		add(new FeatureItem<>(Source.j,  2, Field.lemma));
		
		add(new FeatureItem<>(Source.k,  1, Field.lemma));
		
		// pos features
		add(new FeatureItem<>(Source.i, -2, Field.pos_tag));
		add(new FeatureItem<>(Source.i, -1, Field.pos_tag));
		add(new FeatureItem<>(Source.i,  0, Field.pos_tag));
		add(new FeatureItem<>(Source.i,  1, Field.pos_tag));
		add(new FeatureItem<>(Source.i,  2, Field.pos_tag));
		
		add(new FeatureItem<>(Source.j, -2, Field.pos_tag));
		add(new FeatureItem<>(Source.j, -1, Field.pos_tag));
		add(new FeatureItem<>(Source.j,  0, Field.pos_tag));
		add(new FeatureItem<>(Source.j,  1, Field.pos_tag));
		add(new FeatureItem<>(Source.j,  2, Field.pos_tag));
		
		add(new FeatureItem<>(Source.k,  1, Field.pos_tag));
		add(new FeatureItem<>(Source.k,  2, Field.pos_tag));
		
		// valency features
		add(new FeatureItem<>(Source.i, 0, Field.valency, Direction.all));
		add(new FeatureItem<>(Source.j, 0, Field.valency, Direction.all));
		
		//word form
		add(new FeatureItem<>(Source.i, -1, Field.simplified_word_form));
		add(new FeatureItem<>(Source.i,  0, Field.simplified_word_form));
		add(new FeatureItem<>(Source.i,  1, Field.simplified_word_form));
		
		add(new FeatureItem<>(Source.j, -1, Field.simplified_word_form));
		add(new FeatureItem<>(Source.j,  0, Field.simplified_word_form));
		add(new FeatureItem<>(Source.j,  1, Field.simplified_word_form));
		
		add(new FeatureItem<>(Source.k,  1, Field.simplified_word_form));
		
		//suffix and prefix
		add(new FeatureItem<>(Source.i,  0, Field.suffix));
		add(new FeatureItem<>(Source.j,  0, Field.suffix));
		add(new FeatureItem<>(Source.k,  0, Field.suffix));

		add(new FeatureItem<>(Source.j, 0, Field.prefix));
		add(new FeatureItem<>(Source.i, 0, Field.prefix));
		add(new FeatureItem<>(Source.k, 0, Field.prefix));

		// 2nd-order features
		add(new FeatureItem<>(Source.i, Relation.h  , 0, Field.lemma));
		add(new FeatureItem<>(Source.i, Relation.lmd, 0, Field.lemma));
		add(new FeatureItem<>(Source.i, Relation.rmd, 0, Field.lemma));
		add(new FeatureItem<>(Source.j, Relation.lmd, 0, Field.lemma));
		
		add(new FeatureItem<>(Source.i, Relation.h  , 0, Field.pos_tag));
		add(new FeatureItem<>(Source.i, Relation.lmd, 0, Field.pos_tag));
		add(new FeatureItem<>(Source.i, Relation.rmd, 0, Field.pos_tag));
		add(new FeatureItem<>(Source.j, Relation.lmd, 0, Field.pos_tag));
		
		add(new FeatureItem<>(Source.i,               0, Field.dependency_label));
		add(new FeatureItem<>(Source.i, Relation.lns, 0, Field.dependency_label));
		add(new FeatureItem<>(Source.i, Relation.lmd, 0, Field.dependency_label));
		add(new FeatureItem<>(Source.i, Relation.rmd, 0, Field.dependency_label));
		add(new FeatureItem<>(Source.j, Relation.lmd, 0, Field.dependency_label));
		
		// 3rd-order features
		add(new FeatureItem<>(Source.i, Relation.h2  , 0, Field.lemma));
		add(new FeatureItem<>(Source.i, Relation.lmd2, 0, Field.lemma));
		add(new FeatureItem<>(Source.i, Relation.rmd2, 0, Field.lemma));
		add(new FeatureItem<>(Source.j, Relation.lmd2, 0, Field.lemma));
		
		add(new FeatureItem<>(Source.i, Relation.h2  , 0, Field.pos_tag));
		add(new FeatureItem<>(Source.i, Relation.lmd2, 0, Field.pos_tag));
		add(new FeatureItem<>(Source.i, Relation.rmd2, 0, Field.pos_tag));
		add(new FeatureItem<>(Source.j, Relation.lmd2, 0, Field.pos_tag));
		
		add(new FeatureItem<>(Source.i, Relation.h   , 0, Field.dependency_label));
		add(new FeatureItem<>(Source.i, Relation.lns2, 0, Field.dependency_label));
		add(new FeatureItem<>(Source.i, Relation.lmd2, 0, Field.dependency_label));
		add(new FeatureItem<>(Source.i, Relation.rmd2, 0, Field.dependency_label));
		add(new FeatureItem<>(Source.j, Relation.lmd2, 0, Field.dependency_label));
		
		add(new FeatureItem<>(Source.i, Relation.h, 0, Field.simplified_word_form));
		add(new FeatureItem<>(Source.i, Relation.h2, 0, Field.simplified_word_form));

//2 gram
		//add(new FeatureItem<>(Source.i, Relation.h, 0, Field.lemma), new FeatureItem<>(Source.i, Relation.h2, 0, Field.lemma));
		//add(new FeatureItem<>(Source.i, Relation.h, 0, Field.pos_tag), new FeatureItem<>(Source.i, Relation.h2, 0, Field.pos_tag));
		
		add(new FeatureItem<>(Source.k, 0, Field.simplified_word_form), new FeatureItem<>(Source.k, 1, Field.pos_tag));
		add(new FeatureItem<>(Source.k, 0, Field.prefix), new FeatureItem<>(Source.k, 1, Field.prefix));
		add(new FeatureItem<>(Source.k, 0, Field.suffix), new FeatureItem<>(Source.k, 1, Field.suffix));
		add(new FeatureItem<>(Source.k, 0, Field.lemma), new FeatureItem<>(Source.k, 1, Field.lemma));
//3 gram
		add(new FeatureItem<>(Source.k, 0, Field.lemma), new FeatureItem<>(Source.k, 1, Field.lemma), new FeatureItem<>(Source.k, 2, Field.lemma));
		add(new FeatureItem<>(Source.k, 0, Field.simplified_word_form), new FeatureItem<>(Source.k, 1, Field.pos_tag), new FeatureItem<>(Source.k, 2, Field.pos_tag));
		
		// boolean features
		addSet(new FeatureItem<>(Source.i, 0, Field.binary));
		addSet(new FeatureItem<>(Source.j, 0, Field.binary));
	}
}

