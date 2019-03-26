import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.spark_project.guava.collect.Iterables;

import scala.Tuple2;

public class SparkPageRank {

	private static final String LINK_URI = "file:///f:/my talks and teaching/New folder/examples/datasets/pageLink.txt";
	private static final int totalIteration = 10;
	private static final double beta = 0.85;
	
	public static void main(String[] args) {
		

		// initializing spark
		SparkConf conf = new SparkConf().setAppName("SparkAverage").setMaster("local[2]");
		JavaSparkContext sc = new JavaSparkContext(conf);
		sc.setLogLevel("WARN");
		
		// identify all the neighbors for each page
		class PageNeighbor implements PairFunction<String,String,String> { 		
			@Override
			public Tuple2<String,String> call(String line) throws Exception {
				String[] link = line.split(" ");
				return new Tuple2<String,String>(link[0],link[1]);
			}			
		}
		JavaRDD<String> linkFile = sc.textFile(LINK_URI);
		JavaPairRDD<String, Iterable<String>> links = linkFile.mapToPair( new PageNeighbor() ).distinct().groupByKey().cache();
		System.out.println("links has [" + links.count() + "] elements");
		System.out.println(links.take((int)links.count()).toString());		
		
		// get the total number of pages in the network
		Broadcast<Long> numOfPages = sc.broadcast(links.count());
		
		// initialize the pageRank vector, each component has 1/numOfPages at its initial rank
		JavaPairRDD<String,Double> pageRank = links.mapValues( new Function<Iterable<String>, Double>() {
			@Override
			public Double call(Iterable<String> neighborURL) {
				return 1.0/numOfPages.value();
				// return 1.0;
			}
		} );
		System.out.println("pageRank has [" + pageRank.count() + "] elements");
		System.out.println(pageRank.take((int)pageRank.count()).toString());
		
		// define helper classes
		class RankContribution implements PairFlatMapFunction<Tuple2<Iterable<String>, Double>,String,Double> { 			
			@Override
			public Iterator<Tuple2<String, Double>> call(Tuple2<Iterable<String>, Double> linkConfig) throws Exception {
				List<Tuple2<String,Double>> results = new ArrayList<Tuple2<String,Double>>();
				int neighborCount = Iterables.size(linkConfig._1);
				for (String neighborURL : linkConfig._1) {
					results.add(new Tuple2<String,Double>(neighborURL,linkConfig._2/neighborCount));
				}
				return results.iterator();
			}
			
		}
		class RankAdjust implements Function2<Double,Double,Double> {
			@Override
			public Double call(final Double value1,final Double value2) {
				return value1 + value2;
			}
		}
		
		for ( int i = 0 ; i < totalIteration; i ++ ) {
			
			JavaPairRDD<String, Tuple2<Iterable<String>, Double>> joinedRDD = links.join(pageRank);
			//System.out.println("joinedRDD has [" + joinedRDD.count() + "] elements");
			//System.out.println(joinedRDD.take((int)joinedRDD.count()).toString());
		
			JavaRDD<Tuple2<Iterable<String>, Double>> weightRDD = joinedRDD.values();
			//System.out.println("weightRDD has [" + weightRDD.count() + "] elements");
			// System.out.println(weightRDD.take((int)weightRDD.count()).toString());
		
			// calculate contribution
			JavaPairRDD<String, Double> contribs = weightRDD.flatMapToPair(new RankContribution());
			//System.out.println("contribs has [" + contribs.count() + "] elements");
			//System.out.println(contribs.take((int)contribs.count()).toString());
		
			// addjust current rank
			pageRank = contribs.reduceByKey(new RankAdjust());
			System.out.println("pageRank has [" + pageRank.count() + "] elements");
			System.out.println(pageRank.take((int)pageRank.count()).toString());
	
		}
		System.out.println("pageRank has altogether [" + pageRank.count() + "] elements");
		System.out.println(pageRank.take((int)pageRank.count()).toString());
		// pageRank.saveAsTextFile("pageRankResult.txt");
		
		numOfPages.unpersist();
		sc.close();
		
	}

}