<a id='0'></a>
## Accidents of London 2019 - Analysis
<p>
<div>In this project, I have connected to the Transport of London unified API using a Nifi flow in order to download the public information of the accidents in the city of London in 2019. This Nifi flow ingested a series of JSON files with the accidents information in Hadoop HDFS. After that, we used this Python code in order to create a DataFrame and obtain some insights on it using Spark.</div>
</br>
<div>The main objectives of this code are:</div>
</br>
<ul>
    <li>Create a DataFrame from multiple JSON files in multiple directories.</li>
    <li>Deal with complex data structures such as arrays in JSON documents.</li>
    <li>Provide insights on the created DataFrames.</li>
</ul>    
</p>
