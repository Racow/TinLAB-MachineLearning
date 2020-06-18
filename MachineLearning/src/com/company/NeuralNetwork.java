package com.company;

import org.encog.ConsoleStatusReportable;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.randomize.ConsistentRandomizer;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.missing.MeanMissingHandler;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.encog.util.simple.EncogUtility;
import org.encog.util.simple.TrainingSetUtil;

import java.io.*;
import java.util.Arrays;

public class NeuralNetwork implements Serializable {
    public static final String FILENAME = "encogexample(1.0).eg";

    public int naam = 3;

    public void train() {
        try {

            File file = new File("train_data/aalborg.csv");

            VersatileDataSource source = new CSVDataSource(file, true, CSVFormat.ENGLISH);
            VersatileMLDataSet data = new VersatileMLDataSet(source);
            data.getNormHelper().setFormat(CSVFormat.ENGLISH);

            ColumnDefinition ACCELERATE = data.defineSourceColumn("ACCELERATE", 0, ColumnType.ordinal);
            ACCELERATE.defineClass(new String[]{"0.0", "1.0"});

            ColumnDefinition BRAKE = data.defineSourceColumn("BRAKE", 1, ColumnType.continuous);

            ColumnDefinition STEERING = data.defineSourceColumn("STEERING", 2, ColumnType.continuous);

            data.defineSourceColumn("INPUT1", 3, ColumnType.continuous);
            data.defineSourceColumn("INPUT2", 4, ColumnType.continuous);
            data.defineSourceColumn("INPUT3", 5, ColumnType.continuous);
            data.defineSourceColumn("INPUT4", 6, ColumnType.continuous);
            data.defineSourceColumn("INPUT5", 7, ColumnType.continuous);
            data.defineSourceColumn("INPUT6", 8, ColumnType.continuous);
            data.defineSourceColumn("INPUT7", 9, ColumnType.continuous);
            data.defineSourceColumn("INPUT8", 10, ColumnType.continuous);
            data.defineSourceColumn("INPUT9", 11, ColumnType.continuous);
            data.defineSourceColumn("INPUT10", 12, ColumnType.continuous);
            data.defineSourceColumn("INPUT11", 13, ColumnType.continuous);
            data.defineSourceColumn("INPUT12", 14, ColumnType.continuous);
            data.defineSourceColumn("INPUT13", 15, ColumnType.continuous);
            data.defineSourceColumn("INPUT14", 16, ColumnType.continuous);
            data.defineSourceColumn("INPUT15", 17, ColumnType.continuous);
            data.defineSourceColumn("INPUT16", 18, ColumnType.continuous);
            data.defineSourceColumn("INPUT17", 19, ColumnType.continuous);
            data.defineSourceColumn("INPUT18", 20, ColumnType.continuous);
            data.defineSourceColumn("INPUT18", 21, ColumnType.continuous);
            data.defineSourceColumn("INPUT20", 22, ColumnType.continuous);
            data.defineSourceColumn("INPUT21", 23, ColumnType.continuous);
            data.defineSourceColumn("INPUT22", 24, ColumnType.continuous);

            data.analyze();

            ColumnDefinition outPut[] = new ColumnDefinition[3];
            outPut[0] = BRAKE;
            outPut[1] = ACCELERATE;
            outPut[2] = STEERING;
            data.defineMultipleOutputsOthersInput(outPut);

            EncogModel model = new EncogModel(data);
            model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);
            model.setReport(new ConsoleStatusReportable());

            model.setReport(new ConsoleStatusReportable());

            data.normalize();

            model.holdBackValidation(0.3, true, 1001);

            model.selectTrainingType(data);

            MLRegression bestMethod = (MLRegression) model.crossvalidate(10, true);

            System.out.println("Training error: " + model.calculateError(bestMethod, model.getTrainingDataset()));
            System.out.println("Validation error: " + model.calculateError(bestMethod, model.getValidationDataset()));

            NormalizationHelper helper = data.getNormHelper();
            System.out.println(helper.toString());

            System.out.println("Final model: " + bestMethod);
            EncogDirectoryPersistence.saveObject(new File(FILENAME), bestMethod);

            file.delete();
            Encog.getInstance().shutdown();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public void train2() {
        final String FILENAME = "encogexample.eg";
        BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File(FILENAME));

        // Set TrainingSet
        final MLDataSet trainingSet = TrainingSetUtil.loadCSVTOMemory(CSVFormat.ENGLISH, "train_data/aalborg.csv", true, 22, 3); // toevoegen Inputsize - Idealsize?

        // Train the NN
        int epoch = 1;
        final MLTrain train = new ResilientPropagation(network, trainingSet);
        do {
            epoch++;
            train.iteration();
            if (epoch % 10 == 0) {
                System.out.println(train.getError());
            }
        } while (train.getError() > 20);

        // print Training
        double error = network.calculateError(trainingSet);
        System.out.println("Network training error: " + error);
        EncogDirectoryPersistence.saveObject(new File("encogexample(2).eg"), network);
    }

    public void train3() {
        final String FILENAME = "encogexample(1.3).eg";
        BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File(FILENAME));

        File file = new File("train_data/aalborg.csv");

        VersatileDataSource source = new CSVDataSource(file, true, CSVFormat.ENGLISH);
        VersatileMLDataSet data = new VersatileMLDataSet(source);
        data.getNormHelper().setFormat(CSVFormat.ENGLISH);

        ColumnDefinition ACCELERATE = data.defineSourceColumn("ACCELERATE", 0, ColumnType.ordinal);
        ACCELERATE.defineClass(new String[]{"0.0", "1.0"});

        ColumnDefinition BRAKE = data.defineSourceColumn("BRAKE", 1, ColumnType.continuous);

        ColumnDefinition STEERING = data.defineSourceColumn("STEERING", 2, ColumnType.continuous);

        data.defineSourceColumn("INPUT1", 3, ColumnType.continuous);
        data.defineSourceColumn("INPUT2", 4, ColumnType.continuous);
        data.defineSourceColumn("INPUT3", 5, ColumnType.continuous);
        data.defineSourceColumn("INPUT4", 6, ColumnType.continuous);
        data.defineSourceColumn("INPUT5", 7, ColumnType.continuous);
        data.defineSourceColumn("INPUT6", 8, ColumnType.continuous);
        data.defineSourceColumn("INPUT7", 9, ColumnType.continuous);
        data.defineSourceColumn("INPUT8", 10, ColumnType.continuous);
        data.defineSourceColumn("INPUT9", 11, ColumnType.continuous);
        data.defineSourceColumn("INPUT10", 12, ColumnType.continuous);
        data.defineSourceColumn("INPUT11", 13, ColumnType.continuous);
        data.defineSourceColumn("INPUT12", 14, ColumnType.continuous);
        data.defineSourceColumn("INPUT13", 15, ColumnType.continuous);
        data.defineSourceColumn("INPUT14", 16, ColumnType.continuous);
        data.defineSourceColumn("INPUT15", 17, ColumnType.continuous);
        data.defineSourceColumn("INPUT16", 18, ColumnType.continuous);
        data.defineSourceColumn("INPUT17", 19, ColumnType.continuous);
        data.defineSourceColumn("INPUT18", 20, ColumnType.continuous);
        data.defineSourceColumn("INPUT18", 21, ColumnType.continuous);
        data.defineSourceColumn("INPUT20", 22, ColumnType.continuous);
        data.defineSourceColumn("INPUT21", 23, ColumnType.continuous);
        data.defineSourceColumn("INPUT22", 24, ColumnType.continuous);

        data.analyze();



        ColumnDefinition outPut[] = new ColumnDefinition[3];
        outPut[0] = BRAKE;
        outPut[1] = ACCELERATE;
        outPut[2] = STEERING;
        data.defineMultipleOutputsOthersInput(outPut);

        EncogModel model = new EncogModel(data);
        model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);


        data.normalize();

        //model.holdBackValidation(0.3, true, 1001);

        model.selectTrainingType(data);

        (new ConsistentRandomizer(-1, 1, 100)).randomize(network);

        // train the neural network
        final MLTrain train = new ResilientPropagation(network, data);
        int count = 0;
        do {
            train.iteration();
            count++;
            if (count % 10000 == 0){
                System.out.println(train.getError());
                EncogDirectoryPersistence.saveObject(new File("encogexample(1.3).eg"), network);
            }
        } while (train.getError() > 0.0);
        Encog.getInstance().shutdown();
    }
}