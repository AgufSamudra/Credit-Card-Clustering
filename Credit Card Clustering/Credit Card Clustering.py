#!/usr/bin/env python
# coding: utf-8

# # Deskripsi Data
# 
# Hal ini membutuhkan pengembangan segmentasi pelanggan untuk menentukan strategi pemasaran. Sampel
# Dataset merangkum perilaku penggunaan sekitar 9000 pemegang kartu kredit aktif selama 6 bulan terakhir. File berada pada level pelanggan dengan 18 variabel perilaku.
# 
# * `CUST ID` : Identifikasi pemegang Kartu Kredit (Kategoris)
# * `BALANCE` : Jumlah saldo yang tersisa di akun mereka untuk melakukan pembelian
# * `BALANCEFREQUENCY` : Seberapa sering Saldo diperbarui, skor antara 0 dan 1 (1 = sering diperbarui, 0 = tidak sering diperbarui)
# * `PURCHASES` : Jumlah pembelian yang dilakukan dari akun
# * `ONEOFFPURCHASES` : Jumlah pembelian maksimum dilakukan dalam sekali jalan
# * `INSTALLMENTSPURCHASES` : Jumlah pembelian dilakukan secara angsuran
# * `CASHADVANCE` : Uang tunai di muka yang diberikan oleh pengguna
# * `PURCHASESFREQUENCY` : Seberapa sering Pembelian dilakukan, skor antara 0 dan 1 (1 = sering dibeli, 0 = tidak sering dibeli)
# * `ONEOFFPURCHASESFREQUENCY` : Seberapa sering Pembelian terjadi dalam sekali jalan (1 = sering dibeli, 0 = tidak sering dibeli)
# * `PURCHASESINSTALLMENTSFREQUENCY` : Seberapa sering pembelian secara mencicil dilakukan (1 = sering dilakukan, 0 = tidak sering dilakukan)
# * `CASHADVANCEFREQUENCY` : Seberapa sering uang tunai di muka dibayarkan
# * `CASHADVANCETRX` : Jumlah Transaksi yang dilakukan dengan "Cash in Advanced"
# * `PURCHASESTRX` : Jumlah transaksi pembelian yang dilakukan
# * `CREDITLIMIT` : Batas Kartu Kredit untuk pengguna
# * `PAYMENTS` : Jumlah Pembayaran yang dilakukan oleh pengguna
# * `MINIMUM_PAYMENTS` : Minimum amount of payments made by user
# * `PRCFULLPAYMENT` : Jumlah minimum pembayaran yang dilakukan oleh pengguna
# * `TENURE` : Jangka waktu layanan kartu kredit untuk pengguna

# # Import Library

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # Import Dataset

# In[2]:


df = pd.read_csv('dataset/CC GENERAL.csv')
df


# Kita akan melihat info dari dataset kita dengan fungsi `info()`. Dan di dapati `14` tipe data **float**, `3` tipe data **integer**, `1` tipe data **object**

# In[3]:


df.info()


# # Exploratory Data Analysis
# Kita membuat varibale baru bernama `num` untuk membuang varibel **CUST_ID**, variable `num` hanya di gunakan untuk EDA (Exploratory Data Analysis)

# In[4]:


num = df.drop('CUST_ID', axis=1)
num.head()


# In[5]:


df


# Kemudian kita mencari korelasi antar variabel dengan fungsi `corr()` kita bisa melihat hasilnya di bawah, namun hasil akan lebih mudah terlihat jika menggunakan visualisasi.

# In[6]:


cor= num.corr(method='pearson')
print(cor)


# Dengan `heatmap` kita bisa melihat visual dari korelasi antar varibel dalam dataset.

# In[7]:


plt.figure(figsize=(10, 8))
correlation_matrix = num.corr().round(6)
 
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix", size=20)


# In[8]:


df


# Selanjutnya kita akan mengecek apakah data memiliki Missing Value di dalamnya menggunakan `isnull().sum()`

# In[9]:


df.isnull().sum()


# Jika di lihat pada data di atas, hanya kolom `MINUMUM_PAYMENTS` dan `CREDIT_LIMIT` yang memiliki missing value. Di sini saya memutuskan untuk mengisi nilai kosong tersebut dengan nilai mean.

# In[10]:


df['MINIMUM_PAYMENTS'].mean()


# In[11]:


df['MINIMUM_PAYMENTS'].fillna(value=df['MINIMUM_PAYMENTS'].mean(), inplace=True)


# In[12]:


df['CREDIT_LIMIT'].mean()


# In[13]:


df['CREDIT_LIMIT'].fillna(value=df['CREDIT_LIMIT'].mean(), inplace=True)


# Setelah selesai melakukan pengisian nilai kosong dengan `fillna()` data pada dataset kita sudah bersih dari missing value atau nilai kosong.

# In[14]:


df.isnull().sum()


# Kita juga akan melihat deskripsi dataset menggunakan `describe()`

# In[15]:


df.describe()


# Serta menghapus kolom `CUST_ID` dari dataset, karena kita tidak terlalu membutuhkan nya. Oleh karena itu kolom tersebut akan kita drop, dengan `drop()`

# In[16]:


df.drop(['CUST_ID'], axis=1, inplace=True)
df.head()


# Dan merubah variable `df` menjadi `X`.

# In[17]:


X = df
X.head()


# # Data Preparation

# Pertama kita akan melakukan Standarisasi untuk semua data pada dataset kita, yang kita gunakan adalah `StandardScaler` dari Scikit-Learn

# In[18]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[19]:


X


# Pada kasus kita, kita menggunakan Algoritma `K-Means` namun sebelumnya kita akan mencoba mecari `n_clusters` yang sesuai dengan melakukan perulangan.

# In[20]:


from sklearn.cluster import KMeans
 
#membuat list yang berisi inertia
clusters = []
for i in range(1,11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42).fit(X)
    clusters.append(km.inertia_)


# Setelah itu kita akan menvisualisasikan hasil perulangan sekaligus memberikan output `inertia` atau nilai `n_clusters` yang sesuai dengan kasus kita. Dan di dapati 5 untuk kasus kita kali ini.

# In[21]:


# membuat plot inertia
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Cari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')


# Sesudah mencari n_clusters yang sesuai, kita akan mencoba mereduksi semua fitur kita pada dataset.

# In[22]:


from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)
pca.explained_variance_ratio_


# In[23]:


plt.figure(figsize=(12,10))
plt.plot(range(1, 18), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--',)
plt.title('PCA Captured Variance')
plt.xlabel('# Principal Components')
plt.ylabel('Captured Variance')
plt.show()


# Kita memutuskan menggunakan `n_component = 2` untuk melakukan PCA terhadap dataset, atau mereduksi semua fitur yang ada.

# In[24]:


pca = PCA(n_components=2)
pca.fit(X)
pca.components_


# In[25]:


pca_scores = pca.fit_transform(X)
pca_scores


# # Model Development

# Kita membuat dataframe baru bernama `pca_df` dengan data yang berasal dari `pca_scores` serta membuat kolom baru bernama `pca1` dan `pca2`. Alasannya adalah karena sebelumnya kita mengatur `n_component` sebesar **2**, maka columns yang kita buat menjadi **2**

# In[26]:


pca_df = pd.DataFrame(data = pca_scores, columns=['pca1', 'pca2'])
pca_df.head()


# ## KMeans

# Lalu kita mulai membuat objek km5 dari class KMeans dan beberapa parameter yang di gunakan. Serta kita juga meng `copy()` data `pca_df` dan di masukan ke dalam variable baru bernama `data_kmeans`, lalu membuat kolom baru bernama `Labels` yang di isi dengan `km5` (variable objek KMeans)

# In[27]:


# membuat objek KMeans
km5 = KMeans(n_clusters=5, init='k-means++', random_state=42).fit(pca_df)


# menambahkan kolom label pada dataset
data_kmeans = pca_df.copy()
data_kmeans['Labels'] = km5.labels_


# In[28]:


data_transf_kmeans = data_kmeans.groupby('Labels').mean()
data_transf_kmeans


# # Visualization Result

# Terakhir kita akan melihat hasil dari Clustering dengan menggunakan Visualisasi, kita bisa lihat hasil yang di berikan cukup baik. Dan kita sudah memberikan label kepada data yang kita punya, dan melakukan visualisasi terhadap data yang telah di proses dengan melakukan `Standarisasi` dan `PCA`.

# In[30]:


# membuat plot KMeans dengan 5 klaster
plt.figure(figsize=(20,10))
sns.scatterplot(pca_df['pca1'], pca_df['pca2'], hue=data_kmeans['Labels'],
                palette=sns.color_palette('hls', 5))
plt.title('KMeans dengan 5 Cluster')
plt.show()


# In[ ]:




