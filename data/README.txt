Datasets for data integration:

The datasets contain 2741 aligned patients. 
The genetic features are 347
The vampire features are 157
The clinical features are 15

The possible outputs to predict are 4. 

The datasets may require imputing. Pay attention. 



- dataset_genetic.csv contains the genomic features
	All the genetic features start with "rs" followed by a number except three columns that are
	bp_gs, alzGS, cvdGS that are respectively composite genes scores for blood pressure, 
	alzheimer and CVD
- dataset_vampire.csv contains the engineered features
- dataset_clinical.csv contains the clinical features that are respectively:
	x gen: gender
        x dur_diab: duration of diabetes
	x e_age: age at the developmenet of the disease
	x pre: whether or not the subject had heart problems before
	x dbp: distolic blood pressure
	x sbp: systolic blood pressure
	x c_dpb: a mean over the monitoring period
	x c_spb: a mean over the monitoring period
	x ther: therapy the patients is taking, you need to transform this columns using a one-hot 
		encoder
	x gh: glucose
	x chol: cholesterole
	x hdl: High-Density Lipoprotein ("good cholesterole)
	x trig: triglycerides
	x eversmoker 
	x e4: Apoe4 presence(which is linked with dementia) you need to transform this column with
		one-hot encoding or substitute the 2s with 1s.
- outputs.csv contains four columns: 
	x cvd_fail: wheather or not the patien had a cardiovascular failure (heart attack)
	x cvd_time_age: at what age the cardiovascular failure happened
	x dement_fail: wheater or not the patient had an episode of dementia
	x dement_time_age: at what age the dementia episode happened


