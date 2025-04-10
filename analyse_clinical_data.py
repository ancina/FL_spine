import pandas as pd
import os
import matplotlib.pyplot as plt
from config import Config
import seaborn as sns


def extract_patient_ids(root_dir):
	img_dir = os.path.join(root_dir, 'images')
	all_images = [os.path.splitext(f)[0] for f in os.listdir(img_dir)
	              if f.endswith('lat.jpg') or f.endswith('lat.png')]
	all_images = sorted(all_images)
	
	df = pd.DataFrame(index=all_images)
	# filter patients of the centers
	df = df[df.index.str.startswith(('MAD', 'BCN', 'BOR', 'IST'))]
	
	# Extract patient and center identifiers
	df['patient_id'] = df.index
	df["patient_id_base"] = df["patient_id"].str.split("_", n=1).str[0]
	
	n_patients = pd.DataFrame(df['patient_id_base'].unique(), columns=['id'])
	n_patients.index = n_patients['id']
	n_patients['center'] = n_patients.index.str[:3]
	return df, n_patients


def read_clinical_data(path_clinical_data, sheet):
	df = pd.read_excel(path_clinical_data, sheet_name=sheet)
	return df


def extract_patients(df1, df2):
	cols_of_interest = ['Birthday', 'Gender', 'ESSG Diagnosis'] + [col for col in df2.columns if 'BMI' in col]
	
	df_merged = df1.join(
		df2.set_index('Code of the patient')[cols_of_interest],
		on='patient_id_base',
		how='left'
	)
	return df_merged


def extract_info(df_merged):
	df_merged['date_of_xray_str'] = df_merged['patient_id'].str.split('_', n=2).str[1]
	df_merged['date_of_xray'] = pd.to_datetime(df_merged['date_of_xray_str'], format='%d.%m.%Y', errors='coerce')
	df_merged['Birthday'] = pd.to_datetime(df_merged['Birthday'], errors='coerce')
	
	df_merged['Age'] = (df_merged['date_of_xray'] - df_merged['Birthday']).dt.days / 365.25
	df_merged['Age'] = df_merged['Age'].astype(int)
	
	df_merged['center'] = df_merged.index.str[:3]
	
	return df_merged


def plot_age_distribution(df_merged):
	centers = df_merged['center'].unique()
	
	
	
	for center in centers:
		plt.figure(figsize=(12, 8))
		subset = df_merged[df_merged['center'] == center]
		sns.histplot(data = subset['Age'], bins = 100, kde = True)
	
		plt.xlabel('Age')
		plt.ylabel('Frequency')
		plt.title('Age Distribution per Center')
		plt.grid(True)
		plt.savefig(f'age distr {center}')


def generate_report(df_merged, output_filename="report.txt"):
	report_lines = []
	
	report_lines.append("==== OVERALL STATISTICS ====\n")
	
	age_stats = df_merged['Age'].describe()
	report_lines.append("Overall Age Statistics:")
	report_lines.append(age_stats.to_string())
	
	gender_counts = df_merged['Gender'].value_counts()
	gender_pct = df_merged['Gender'].value_counts(normalize=True) * 100
	report_lines.append("\nOverall Gender Counts:")
	report_lines.append(gender_counts.to_string())
	report_lines.append("\nOverall Gender Percentages (%):")
	report_lines.append(gender_pct.to_string())
	
	pathology_counts = df_merged['ESSG Diagnosis'].value_counts()
	pathology_pct = df_merged['ESSG Diagnosis'].value_counts(normalize=True) * 100
	report_lines.append("\nOverall Pathology Counts:")
	report_lines.append(pathology_counts.to_string())
	report_lines.append("\nOverall Pathology Percentages (%):")
	report_lines.append(pathology_pct.to_string())
	
	bmi_columns = [col for col in df_merged.columns if 'BMI' in col]
	bmi_all = df_merged[bmi_columns].melt(value_name='BMI').dropna()
	bmi_all['BMI'] = bmi_all['BMI'].astype(float)
	
	bmi_stats = bmi_all['BMI'].describe()
	report_lines.append("\nOverall BMI Statistics:")
	report_lines.append(bmi_stats.to_string())
	
	report_lines.append("\n==== CENTER-WISE STATISTICS ====\n")
	
	age_center_stats = df_merged.groupby('center')['Age'].describe()
	report_lines.append("Center-wise Age Statistics:")
	report_lines.append(age_center_stats.to_string())
	
	gender_center_counts = df_merged.groupby('center')['Gender'].value_counts()
	gender_center_pct = df_merged.groupby('center')['Gender'].value_counts(normalize=True) * 100
	report_lines.append("\nCenter-wise Gender Counts:")
	report_lines.append(gender_center_counts.to_string())
	report_lines.append("\nCenter-wise Gender Percentages (%):")
	report_lines.append(gender_center_pct.to_string())
	
	pathology_center_counts = df_merged.groupby('center')['ESSG Diagnosis'].value_counts()
	pathology_center_pct = df_merged.groupby('center')['ESSG Diagnosis'].value_counts(normalize=True) * 100
	report_lines.append("\nCenter-wise pathology Counts:")
	report_lines.append(pathology_center_counts.to_string())
	report_lines.append("\nCenter-wise pathology Percentages (%):")
	report_lines.append(pathology_center_pct.to_string())
	
	df_melted_bmi = df_merged.melt(id_vars=['center'], value_vars=bmi_columns, value_name='BMI')
	df_melted_bmi = df_melted_bmi.dropna(subset=['BMI'])
	df_melted_bmi['BMI'] = df_melted_bmi['BMI'].astype(float)
	bmi_center_stats = df_melted_bmi.groupby('center')['BMI'].describe()
	report_lines.append("\nCenter-wise BMI Statistics:")
	report_lines.append(bmi_center_stats.to_string())
	
	report = "\n".join(report_lines)
	
	with open(output_filename, "w") as file:
		file.write(report)
	
	print(f"Report saved to {output_filename}")
	return report


if __name__ == '__main__':
	root_dir = Config.DATA_DIR
	
	imgs_df, n_patients = extract_patient_ids(root_dir)
	counts_1 = n_patients['center'].value_counts()
	path_clinical_data = 'June 2023 All.xlsx'
	sheet_name = 'Clinical Data'
	
	clinical = read_clinical_data(path_clinical_data=path_clinical_data, sheet=sheet_name)
	
	df = extract_patients(df1=imgs_df, df2=clinical)
	df = extract_info(df_merged=df)
	
	report_text = generate_report(df, "my_data_report.txt")
	
	plot_age_distribution(df)
