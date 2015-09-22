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
package edu.emory.mathcs.nlp.benchmark;

import java.util.Arrays;

import org.junit.Test;

/**
 * @author Jinho D. Choi ({@code jinho.choi@emory.edu})
 */
public class Benchmark
{
	@Test
	public void speed()
	{
		final int iter = 1000;
		long st, et;
		
		double[] f = new double[100000];

		st = System.currentTimeMillis();
		for (int i=0; i<iter; i++)
			Arrays.fill(f, 1);
//			IntStream.range(0, f.length).parallel().forEach(n -> { f[n] = 1; });
		et = System.currentTimeMillis();
		System.out.println(et-st);
	}
}
