package spark10;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.apache.spark.sql.SparkSession;
public class frequentpair {
	public static void main(String[] args) {
	SparkSession spark = SparkSession.builder().config("spark.master","local[*]").getOrCreate();
	JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
	//sc.setLogLevel("WARN");
	JavaRDD<String> data = sc.textFile("file:///C:/Users/dell/Downloads/sample_fpgrowth.txt");

	JavaRDD<List<String>> transactions = data.map(line -> Arrays.asList(line.split(" ")));

	FPGrowth fpg = new FPGrowth()
	  .setMinSupport(0.2)
	  .setNumPartitions(10);
	
	FPGrowthModel<String> model = fpg.run(transactions);

			for (FPGrowth.FreqItemset<String> itemset : model.freqItemsets().toJavaRDD().collect()) {
				  System.out.println("[" + itemset.javaItems() + "], " + itemset.freq());
				}

	double minConfidence = 0.8;
				for (AssociationRules.Rule<String> rule
				  : model.generateAssociationRules(minConfidence).toJavaRDD().collect()) {
				  System.out.println(
				    rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence());
				}
}
}
