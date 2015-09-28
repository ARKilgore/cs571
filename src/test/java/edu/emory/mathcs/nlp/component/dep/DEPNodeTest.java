/**
 * Copyright 2015, Emory University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.emory.mathcs.nlp.component.dep;

import static org.junit.Assert.assertEquals;

import java.io.FileInputStream;
import java.util.List;

import org.junit.Test;

import edu.emory.mathcs.nlp.common.util.Joiner;
import edu.emory.mathcs.nlp.component.util.feature.Field;
import edu.emory.mathcs.nlp.component.util.node.FeatMap;
import edu.emory.mathcs.nlp.component.util.reader.TSVReader;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class DEPNodeTest
{
	@Test
	public void test() throws Exception
	{
		TSVReader<DEPNode> reader = new TSVReader<>(new DEPIndex(1, 2, 3, 4, 5, 6));
		reader.open(new FileInputStream("src/main/resources/dat/wsj_0001.dep"));
		DEPNode[] nodes = reader.next();
		
		// TODO:
		System.out.println(Joiner.join(nodes, "\n"));
	}
	
	@Test
	public void testBasicFields()
	{
		DEPNode node = new DEPNode(1, "Jinho");
		
		assertEquals(1      , node.getID());
		assertEquals("Jinho", node.getWordForm());
		
		node = new DEPNode(1, "Jinho", "jinho", "NNP", new FeatMap("fst=jinho|lst=choi"));
		
		assertEquals(1       , node.getID());
		assertEquals("Jinho" , node.getWordForm());
		assertEquals("jinho" , node.getLemma());
		assertEquals("NNP"   , node.getPOSTag());
		
		assertEquals(null	 , node.getHead());
		assertEquals(null	 , node.getGrandHead());
		assertEquals(null	 , node.getLeftNearestSibling());
		assertEquals(null	 , node.getRightNearestSibling());
		assertEquals(null	 , node.getLeftMostDependent());
		assertEquals(null	 , node.getRightMostDependent());
		
		

		assertEquals("jinho"  , node.getFeat("fst"));
		node.removeFeat("fst");
		assertEquals(null  , node.getFeat("fst"));
		assertEquals("choi", node.getFeat("lst"));
		
		node.putFeat("fst", "Jinho");
		assertEquals("Jinho", node.getFeat("fst"));
	}
	
	@Test
	public void testSetters()
	{
		//create dep nodes 1 through 4
		DEPNode node1 = new DEPNode(1, "He");
		DEPNode node2 = new DEPNode(2, "bought");
		DEPNode node3 = new DEPNode(3, "a");
		DEPNode node4 = new DEPNode(4, "car");
		
		
		//set dependencies
		node2.addDependent(node4, "dobj");
		node2.addDependent(node1, "nsubj");
		node4.addDependent(node3, "det");
		
		//check dep lists, right and left dep lists, grand dep list
		List<DEPNode> listNode2 = node2.getDependentList();
		assertEquals(node1, listNode2.get(0));
		assertEquals(node4, listNode2.get(1));
		
		List<DEPNode> listNode2R = node2.getLeftDependentList();
		assertEquals(node1, listNode2R.get(0));
		
		List<DEPNode> listNode2L = node2.getRightDependentList();
		assertEquals(node4, listNode2L.get(0));
		
		List<DEPNode> listNode4 = node4.getDependentList();
		assertEquals(node3, listNode4.get(0));
		
		List<DEPNode> grandDepList = node2.getGrandDependentList();
		assertEquals(node3, grandDepList.get(0));
		
		//check empty grand dep
		List<DEPNode> grandDepListEmpty = node4.getGrandDependentList();
		assert(grandDepListEmpty.isEmpty());
		
		//test dep heads
		assertEquals(node2, node1.getHead());
		assertEquals(node2, node4.getHead());
		assertEquals(node4, node3.getHead());
		assertEquals(node2, node3.getGrandHead());
		
		//test ancestry path (getPath)
		assertEquals(node4.getPath(node1, Field.word_form), "^car^bought|He" );
		
	}
}
