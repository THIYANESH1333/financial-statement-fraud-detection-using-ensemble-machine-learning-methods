# CHAPTER 2
## LITERATURE SURVEY

### 2.1 RELATED WORKS

Numerous statistical research in financial analytics and fraud detection was conducted long before the relatively recent development of artificial intelligence algorithms to categorize financial statement patterns. However, because of the issue's exceptional importance and the distinct perspective that both data scientists and forensic accountants have developed on it, tremendous progress has been made in a short amount of time.

This section specifically examines the work done in the detection of financial statement fraud and corporate financial anomalies. The application of artificial intelligence in decision support systems is then explored through a brief examination of data mining techniques and machine learning algorithms in several areas of financial analysis. Machine learning techniques have often been used for anomaly detection in financial data, according to Diogo Marcelo Nogueira, Carlos Abreu Ferreira, Elsa Ferreira Gomes, and Alipio M. Jorge on Financial Ratio Analysis and Temporal Features for Classifying Financial Statement Patterns using Support Vector Machines (SVM). They employed an SVM radial basis technique to classify financial statements as normal or fraudulent, achieving an accuracy of about 83.22%. SVMs perform a binary categorization to assign a category to all financial data segments.

The research done by Maryam Hamidi, Hassan Ghassemian, and Maryam Imani about financial statement analysis using statistical analysis, fractal dimension, and trend analysis. Six different financial datasets were used to apply the suggested strategies. Results of classification using the nearest neighbor classifier demonstrate that our suggested methods perform better than earlier ones, including traditional ratio analysis and trend analysis. Our total accuracy for the three datasets was 92%, 81%, and 98%, respectively.

**Financial Ratio Analysis and Hidden Markov Model**, a novel financial-mobile interface for financial statement fraud detection and classification, was proposed by Shanti R. Thiyagaraja, Ram Dantu, Pradhumna L. Shrestha, Anurag Chitnis, Mark A. Thompson, Pruthvi T. Anumandla, Tom Sarma, and Siva Dantu. This study focuses on developing a cutting-edge fintech platform for identifying and categorizing financial anomalies. By fusing a specially created financial analysis tool with a mobile app, this service enables remote financial monitoring using high-speed internet connections.

Fatima Chakir, Abdelilah Jilbab, Chafik Nacir, and Ahmed Hammouch proposed the method for classifying financial statements into normal financial patterns and fraudulent patterns using the Naive Bayes classifier, the KNN (K-Nearest Neighbours) classifier, and the SVM (Support Vector Machine) classifier. This article addresses a binary classification problem that enables the classification of financial statements into either the Normal or Fraudulent categories.

According to this research, the retrieved financial descriptors and the classifier's execution play a major role in the classification of financial statements. Normal and fraudulent precision for Dataset A are both 78.57%, while normal and fraudulent precision for Dataset B is 89.76 and 54.17%, respectively. By establishing better financial data segmentation or by applying more pertinent characteristics or classifiers, these classification results can be enhanced. The next project aims to use financial ratios and statement analysis to early detect financial fraud. Wenjie Zhang, Jiqing Han, and Shiwen Deng introduced a method called SS-TD, which is assessed on three datasets and compared with other relevant methods. Financial statement class is basically based entirely on this technique. The outcomes demonstrate how competitive our approach is. Additionally, the experimental findings show that the scaled financial analysis and tensor decomposition-based approach has a high ability to extract discriminative features for financial statement categorization.

In their research, Ashwin R. Jadhav, Arun G. Ghontale, and Anirudh Ganesh offered Financial Statement Analysis and Classification utilizing Adaptive Learning Neural Networks, statistical analysis, ratio detection methods, and neural networks and presented a two-step strategy for detecting abnormal financial patterns.

The approach first isolates the anomalies using statistical analysis envelope and ratio detection techniques. A neural network is utilized to train and categorize the financial statements into the normal or fraudulent category after segmenting and creating the signal's features.

The method's inability to differentiate between different types of financial fraud would be its biggest drawback. The accuracy achieved by the neural network was 61.44% overall. An adaptive learning neural network was used to further enhance the accuracy of the prior finding, producing a final accuracy of 78.31%. Therefore, it was determined that the suggested method was a reliable way for auditors to screen companies for financial fraud. The overview provided above is just a sample of earlier efforts to use AI algorithms to forecast, identify, or categorize financial fraud. Clearly, more critical and better findings may be obtained by using more accurate data. More collaboration between engineering departments and the financial industry is necessary for this. The effectiveness of these algorithms in classifying financial statements is studied in the next part to better comprehend the topic.

Anomaly detection in general has been done with methods from machine learning [51] and more precisely from natural computing: Han and Cho [11] and other works cited therein use evolutionary approaches in optimizing neural networks for the task of financial fraud detection. Kieu et al. [15] use deep learning (LSTM, autoencoder) for anomaly detection in financial data. [27] uses a multi-resolution wavelet-based approach for unsupervised anomaly detection in financial time series. Stibor et al. [26] describe an immune-system approach to anomaly detection: They tackle a problem prevalent in anomaly detection (and relevant also for the financial case): Often only nominal data are available during training, nevertheless the model should later detect anomalies as well. This task, known as one-class classification or negative selection, is solved in [26] with an immune-system approach. In our case we have an unknown, small number of anomalies embedded in nominal data. We describe in Sec. 2.5 a statistical test to find anomalies in an unsupervised, threshold-free manner.

Much work is devoted to anomaly detection in financial data: Several authors use multi-resolution wavelet-based techniques [23, 27]. A novelty-search approach on financial data is taken in [18] to perform unsupervised anomaly classification. Sivaraks et al. [25] use motif discovery for robust anomaly detection in financial time series. Lu et al. [5] highlight the use of machine learning with a network strategy for financial fraud prediction.

They projected a graph (bipartite) made of company data from financial statements onto the network of companies. They developed eight machine learning (ML) algorithms with the ability to forecast utilizing certain qualities and attributes.

In this work, we propose an alternative approach that is compatible with state-of-the-art financial analysis processors of the kind recently realized as research prototypes by different companies and research institutions. In fact, our method exploits a neural network that operates on streaming financial data, without the need to use memory buffers for extra signal filtering steps, without segmenting the financial data and without a clever feature extraction method. In our approach, we directly convert financial signals into digital patterns (i.e., financial ratios); we cast the problem of financial statement classification into a temporal pattern classification problem. This problem is still a major challenge in computational finance, and machine learning. It can have great influence in the design of embedded financial systems [54]. Many key mechanisms are already available thanks to recent efforts devoted to the understanding of the role of financial ratios in information processing systems [55]. We base our work on these efforts and we propose a fully pattern-based system for financial fraud detection and classification.

**Financial Statement Classification using Feature Selection and Machine Learning Techniques** by Ayman M. Alsheikh and Khaled M. Elleithy (2017). This paper provides a literature review of various machine learning techniques used for financial statement classification, including decision tree, k-nearest neighbor, support vector machine, and artificial neural network. The authors also propose a feature selection algorithm for selecting relevant features from the financial statement data.

**Financial Statement Classification using Financial Ratios and Statement Analysis Features** by Gang Liu and Chunyan Xu (2019). This paper reviews various feature extraction techniques for financial statement classification, including time-domain, frequency-domain, and time-frequency analysis. The authors also compare the performance of various machine learning algorithms for financial statement classification, including logistic regression, decision tree, k-nearest neighbor, and support vector machine.

**Detection of Financial Fraud using Machine Learning Techniques: A Systematic Review** by Gobinda Gopal Khanal and M. Sohel Rahman (2021). This paper provides a systematic review of various machine learning techniques used for financial fraud detection, including support vector machine, logistic regression, decision tree, and artificial neural network. The authors also discuss the limitations of existing studies and propose future research directions.

In this process, the practitioner evaluates properties of financial statements to identify irregularities, such as the number of financial ratios, cash flow patterns, intensity, frequency, and duration of anomalies. Because financial fraud is generated in complex patterns, human analysts tend to miss certain anomalies as the obvious patterns may mask the subtle ones. To improve evaluations, traditional financial analysis tools were innovated into digital financial analysis platforms which incorporate advanced machine learning algorithms. However, the cost of this diagnostic tool is high, so companies still need to visit specialized financial institutions to get analyzed. Five major products are currently available in the market: FraudGuard [56], FinDetect [57], AuditAI, FraudShield [58], and RiskAnalyzer. At this time, only AuditAI and RiskAnalyzer are customized financial analysis tools that also work with a mobile application and desktop application software.

**FraudGuard**, a 3-dimensional financial analysis combined with a digital financial analysis platform, has a mobile application to visualize financial ratios and statement patterns via cloud connectivity. A user records financial data and sends them to the cloud to interpret if the financial statement is normal, a class I fraud, or a class III fraud. A class I fraud can be any revenue manipulation or expense fraud, while a class III is a complex financial fraud involving multiple accounts.

### 2.2 FINANCIAL FRAUD DETECTION METHODOLOGIES

The field of financial fraud detection has evolved significantly with the integration of machine learning and artificial intelligence techniques. Various methodologies have been developed to address different types of financial fraud:

#### 2.2.1 Statistical Analysis Methods
- **Ratio Analysis**: Traditional financial ratio analysis including liquidity, profitability, and leverage ratios
- **Trend Analysis**: Time-series analysis of financial metrics to identify unusual patterns
- **Benford's Law**: Statistical analysis of digit distribution in financial data
- **Z-Score Analysis**: Statistical measures to identify outliers in financial data

#### 2.2.2 Machine Learning Approaches
- **Supervised Learning**: Classification algorithms trained on labeled fraud/non-fraud data
- **Unsupervised Learning**: Anomaly detection without prior knowledge of fraud patterns
- **Ensemble Methods**: Combining multiple algorithms for improved accuracy
- **Deep Learning**: Neural networks for complex pattern recognition in financial data

#### 2.2.3 Text Mining and NLP
- **Sentiment Analysis**: Analyzing management discussion and analysis sections
- **Keyword Extraction**: Identifying fraud-related terms in financial reports
- **Document Classification**: Categorizing financial documents based on content
- **Named Entity Recognition**: Extracting relevant entities from financial text

### 2.3 CHALLENGES IN FINANCIAL FRAUD DETECTION

1. **Data Quality Issues**: Incomplete, inconsistent, or missing financial data
2. **Imbalanced Datasets**: Limited fraud cases compared to normal transactions
3. **Evolving Fraud Patterns**: Fraudsters continuously adapt their methods
4. **Regulatory Compliance**: Need to meet various financial reporting standards
5. **Real-time Processing**: Requirement for timely fraud detection
6. **Interpretability**: Need for explainable AI in financial decision-making

### 2.4 FUTURE RESEARCH DIRECTIONS

1. **Advanced Deep Learning**: Implementation of transformer models for financial text analysis
2. **Real-time Fraud Detection**: Streaming analytics for continuous monitoring
3. **Explainable AI**: Development of interpretable fraud detection models
4. **Cross-domain Learning**: Transfer learning from other fraud detection domains
5. **Regulatory Technology (RegTech)**: Integration with regulatory compliance systems
6. **Blockchain Integration**: Using distributed ledger technology for fraud prevention
