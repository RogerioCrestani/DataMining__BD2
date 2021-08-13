package Diabetes_DataMining;

import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Rogerio Crestani
 */
public class DataMining {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        try {
            ConverterUtils.DataSource ds = new ConverterUtils.DataSource("src\\Diabetes_DataMining\\diabetes.arff");
            Instances ins = ds.getDataSet();
            //System.out.println(ins.toString());
            int sair = 1;
            
            while(sair != 0){
                ins.setClassIndex(8);
                
                NaiveBayes nb = new NaiveBayes();
                nb.buildClassifier(ins);
                
                System.out.println("====Calculadora de possibilidade de diabetes====\n");
                System.out.println("Digite os atributos");
                System.out.print("Número de vezes que engravidou: ");
                float atr1 = scanner.nextFloat();
                System.out.print("Concentração de glicose plasmática a 2 horas em um teste de tolerância à glicose oral: ");
                float atr2 = scanner.nextFloat();
                System.out.print("Pressão arterial diastólica (mm Hg): ");
                float atr3 = scanner.nextFloat();
                System.out.print("Espessura da dobra da pele do tríceps (mm): ");
                float atr4 = scanner.nextFloat();
                System.out.print("Insulina sérica de 2 horas (mu U / ml): ");
                float atr5 = scanner.nextFloat();
                System.out.print("Indice de massa corporal (peso em kg / (altura em m) ^ 2): ");
                float atr6 = scanner.nextFloat();
                System.out.print("Função de pedigree de diabetes: ");
                float atr7 = scanner.nextFloat();
                System.out.print("Idade (anos): ");
                float atr8 = scanner.nextFloat();
                
                Instance novo = new DenseInstance(9);
                novo.setDataset(ins);
                novo.setValue(0, atr1);
                novo.setValue(1, atr2);
                novo.setValue(2, atr3);
                novo.setValue(3, atr4);
                novo.setValue(4, atr5);
                novo.setValue(5, atr6);
                novo.setValue(6, atr7);
                novo.setValue(7, atr8);


                double probabilidade[] = nb.distributionForInstance(novo);

                System.out.println("Resultado:");
                float positivo = (float)(probabilidade[1] * 100);
                float negativo = (float)(probabilidade[0] * 100);
                System.out.println("Positivo: " + String.format("%.2f", positivo) + "%");
                System.out.println("Negativo: " + String.format("%.2f", negativo) + "%");

                System.out.println("Tecle 1 para continuar ou 0 para sair: ");
                sair = scanner.nextInt();
                
            }
            

        } catch (Exception ex) {
            Logger.getLogger(DataMining.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}
