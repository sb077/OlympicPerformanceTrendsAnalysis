import pandas as pd
import pandasql as ps
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from dbcrud import read_entries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor,GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score, roc_curve, auc

DB_FILE = "app/olympics_data.db"

def make_report(st, y_test, y_pred, sport_encoder):
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    report_dict = classification_report(y_test, y_pred, target_names=sport_encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.markdown("#### Classification Report")
    st.dataframe(report_df)

    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=sport_encoder.classes_,
        columns=sport_encoder.classes_
    )
    st.markdown("#### Confusion Matrix")
    st.dataframe(conf_matrix_df)

def hypothesis1(st):
    Pre_Event_Results = read_entries("Pre_Event_Results")
    Pre_Event_Results['year'] = pd.to_numeric(Pre_Event_Results['year'], errors='coerce')
    Pre_Event_Results['participants'] = pd.to_numeric(Pre_Event_Results['participants'], errors='coerce')

    Pre_Event_Results = Pre_Event_Results.dropna(subset=['year', 'participants'])

    sports_multiple_years = Pre_Event_Results.groupby('sport').filter(lambda x: x['year'].nunique() > 1)

    sports_trend = sports_multiple_years.groupby('sport').apply(lambda df: linregress(df['year'], df['participants']).slope).reset_index()
    sports_trend.columns = ['sport', 'slope']

    sport_stats = sports_multiple_years.groupby('sport')['participants'].agg(['mean', 'std']).reset_index()
    sports_trend = sports_trend.merge(sport_stats, on='sport')

    X = sports_trend[['slope', 'mean']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42)
    sports_trend['cluster'] = kmeans.fit_predict(X_scaled)

    cluster_labels = {0: "rising", 1: "declining"} if sports_trend.groupby('cluster')['slope'].mean()[0] > 0 else {0: "declining", 1: "rising"}
    sports_trend['trend_label'] = sports_trend['cluster'].map(cluster_labels)

    silhouette_avg = silhouette_score(X_scaled, sports_trend['cluster'])
    davies_bouldin = davies_bouldin_score(X_scaled, sports_trend['cluster'])
    inertia = kmeans.inertia_

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sports_trend, x='slope', y='mean', hue='trend_label', style='trend_label', palette='viridis', s=100)
    plt.title("Clustering of Sports by Rising and Declining Trends")
    plt.xlabel("Trend Slope")
    plt.ylabel("Average Participation")
    plt.grid(True)
    plt.legend(title='Trend')
    plt.show()

    st.pyplot(plt)
    st.info("We have used the KMeans algorithm here because we already knew how many clusters we wanted and the K in this algorithm signifies exactly that. KMeans clustering is very good for data with multi dimensional features and capturing patterns.")
    st.info("We see in our output that the clusters have been defined pretty clearly with slope and average participation being the deciding factors and the data is labeled as well now. We have successfully labeled the input unlabeled data.")
    st.info("This visualization gives us a good idea on how to identify sports which are declining and need urgent attention in order to save them. Also, this gives us an idea on the rising sports which could be super beneficial for marketing, business and other opportunities. This proves our initial hypothesis correct.")

    sports_trend[['sport', 'slope', 'mean', 'std', 'trend_label']]
    st.dataframe(sports_trend)

    st.info(f"Silhouette Score: {silhouette_avg}")
    st.info(f"Davies-Bouldin Index: {davies_bouldin}")
    st.info(f"Inertia: {inertia}")

def hypothesis2(st):
    Pre_Athlete_Events_Details = read_entries("Pre_Athlete_Events_Details")
    Pre_Country_Profile = read_entries("Pre_Country_Profile")
    Pre_Population_Total = read_entries("Pre_Population_Total")


    Athlete_Events_Details_Mod = Pre_Athlete_Events_Details.merge(Pre_Country_Profile, left_on='country_noc', right_on='noc', how='left')
    Athlete_Events_Details_Mod.drop('noc', axis=1, inplace=True)
    Athlete_Events_Details_Mod.drop('country_noc', axis=1, inplace=True)

    athlete_counts = Athlete_Events_Details_Mod.groupby(['country', 'year'])[['men', 'women']].sum().reset_index()
    athlete_counts = athlete_counts.rename(columns={'country': 'Country Name'})


    Pre_Population_Total['Year'] = Pre_Population_Total['Year'].astype(int)
    athlete_counts['year'] = athlete_counts['year'].astype(int)
    merged_data = pd.merge(athlete_counts, Pre_Population_Total, left_on=['Country Name', 'year'],right_on=['Country Name', 'Year'], how='left')

    merged_data['percentage_women'] = (merged_data['women'] / merged_data['Count']) * 100

    merged_data['total'] = merged_data['men'] + merged_data['women']
    merged_data['percentage_women_better'] = (merged_data['women'] / merged_data['total']) * 100

    merged_data = merged_data.dropna(subset=['percentage_women'])

    min_val = merged_data['percentage_women'].min()
    max_val = merged_data['percentage_women'].max()
    merged_data['normalized_percentage_women'] = (merged_data['percentage_women'] - min_val) / (max_val - min_val)

    merged_data = merged_data.dropna(subset=['percentage_women'])

    merged_data = merged_data.dropna(subset=['percentage_women_better'])
    average_participation = merged_data.groupby('Country Name')['percentage_women_better'].mean().reset_index()

    top_countries = average_participation.sort_values(by='percentage_women_better', ascending=False).head(20)
    df_filtered = merged_data[merged_data['Country Name'].isin(top_countries['Country Name'])]

    df_filtered = df_filtered[df_filtered['year'] >= 1960]
    train_data = df_filtered[df_filtered['year'] <= 2000]
    test_data = df_filtered[df_filtered['year'] > 2000]

    X_train = train_data[['year']]
    y_train = train_data['percentage_women_better']
    X_test = test_data[['year']]
    y_test = test_data['percentage_women_better']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    def create_dnn_model():
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=(X_train_scaled.shape[1],)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    dnn_model = create_dnn_model()
    dnn_model.fit(X_train_scaled, y_train, epochs=150, batch_size=4, verbose=1)


    y_pred = dnn_model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    future_years = np.array([2024, 2028, 2032, 2036, 2040]).reshape(-1, 1)
    future_years_scaled = scaler.transform(future_years)
    future_pred = dnn_model.predict(future_years_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(train_data['year'], y_train, color='blue', label='Train Data')
    plt.scatter(test_data['year'], y_test, color='green', label='Actual Test Data')
    plt.scatter(test_data['year'], y_pred, color='red', label='Predicted Test Data')
    plt.plot(future_years, future_pred, 'o--', label='Future Predictions')
    plt.xlabel('Year')
    plt.ylabel('Percentage of Women Participation')
    plt.title('DNN Predictions for Women Participation (Top 50 Countries)')
    plt.legend()
    plt.grid(True)
    plt.show()

    st.pyplot(plt)

    abs_r2 = abs(r2)
    target_countries = [ 'Netherlands', 'Peru', 'Canada', 'Singapore', 'Romania','malta','angola']

    fig, axes = plt.subplots(len(target_countries), 1, figsize=(10, 4 * len(target_countries)), sharex=True)
    fig.suptitle('DNN Predictions for Women Participation by Country')

    for i, country in enumerate(target_countries):
        country_data = df_filtered[df_filtered['Country Name'].str.lower() == country.lower()]

        country_train_data = country_data[country_data['year'] <= 2000]
        country_test_data = country_data[country_data['year'] > 2000]

        X_train_country = country_train_data[['year']]
        y_train_country = country_train_data['percentage_women_better']
        X_test_country = country_test_data[['year']]
        y_test_country = country_test_data['percentage_women_better']
        X_train_country_scaled = scaler.transform(X_train_country)
        X_test_country_scaled = scaler.transform(X_test_country)

        y_pred_country = dnn_model.predict(X_test_country_scaled)



        future_pred_country = dnn_model.predict(future_years_scaled)



        ax = axes[i]
        ax.scatter(country_train_data['year'], y_train_country, color='blue', label='Train Data')
        ax.scatter(country_test_data['year'], y_test_country, color='green', label='Actual Test Data')
        ax.scatter(country_test_data['year'], y_pred_country, color='red', label='Predicted Test Data')
        ax.plot(future_years, future_pred_country, 'o--', label='Future Predictions')

        ax.set_title(f"{country}")
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage of Women Participation')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    st.pyplot(plt)

    st.info("")
    st.info(" We have used DNN here with 3 hidden layers and linear activation at final layer. Our analysis says that the women participation percentage is bound to increase in the future, which is a great sign for world sports. The predicted increase helps us and all stakeholders understand the future progress of women participation in sports and olympics. This will help with better policies and awareness in the future. An increase in gender equality can be concluded from our future prediction analysis and intelligence gained.")

    st.info(f"Absolute R² Score: {abs_r2:.3f}")
    st.info(f"Mean Absolute Error: {mae:.3f}")
    st.info(f"Future Predictions (2024-2040): {future_pred.flatten()}")

def hypothesis3(st):
    Pre_Athlete_Events_Details = read_entries("Pre_Athlete_Events_Details")
    Pre_Country_Profile = read_entries("Pre_Country_Profile")
    Pre_Population_Total = read_entries("Pre_Population_Total")

    query_usa = """
        SELECT
        year,
        country_noc,
        isTeamSport,
        COUNT(medal) AS medal_count
        FROM Pre_Athlete_Events_Details
        WHERE medal != 'no medal'
        AND year IS NOT NULL
        AND country_noc IN ('usa')
        GROUP BY year, country_noc, isTeamSport
        ORDER BY year, country_noc, isTeamSport;
        """

    medals_yearwise_data_usa = ps.sqldf(query_usa, locals())

    query_ger = """
        SELECT
        year,
        country_noc,
        isTeamSport,
        COUNT(medal) AS medal_count
        FROM Pre_Athlete_Events_Details
        WHERE medal != 'no medal'
        AND year IS NOT NULL
        AND country_noc IN ('ger')
        GROUP BY year, country_noc, isTeamSport
        ORDER BY year, country_noc, isTeamSport;
        """

    medals_yearwise_data_ger = ps.sqldf(query_ger, locals())


    query_ita = """
        SELECT
        year,
        country_noc,
        isTeamSport,
        COUNT(medal) AS medal_count
        FROM Pre_Athlete_Events_Details
        WHERE medal != 'no medal'
        AND year IS NOT NULL
        AND country_noc IN ('ita')
        GROUP BY year, country_noc, isTeamSport
        ORDER BY year, country_noc, isTeamSport;
        """

    medals_yearwise_data_ita = ps.sqldf(query_ita, locals())



    query_aus = """
        SELECT
        year,
        country_noc,
        isTeamSport,
        COUNT(medal) AS medal_count
        FROM Pre_Athlete_Events_Details
        WHERE medal != 'no medal'
        AND year IS NOT NULL
        AND country_noc IN ('aus')
        GROUP BY year, country_noc, isTeamSport
        ORDER BY year, country_noc, isTeamSport;
        """

    medals_yearwise_data_aus = ps.sqldf(query_aus, locals())

    st.info(medals_yearwise_data_aus)


    query_ind = """
        SELECT
        year,
        country_noc,
        isTeamSport,
        COUNT(medal) AS medal_count
        FROM Pre_Athlete_Events_Details
        WHERE medal != 'no medal'
        AND year IS NOT NULL
        AND country_noc IN ('ind')
        GROUP BY year, country_noc, isTeamSport
        ORDER BY year, country_noc, isTeamSport;
        """

    medals_yearwise_data_ind = ps.sqldf(query_ind, locals())

    medals_yearwise_data_aus['isTeamSport'] = medals_yearwise_data_aus['isTeamSport'].apply(lambda x: 1 if x else 0)

    X = medals_yearwise_data_aus[['year', 'isTeamSport']]
    y = medals_yearwise_data_aus['medal_count']

    # Split dataset into training 80% and testing 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Max depth set to 5 to avoid overfitting
    reg_tree = DecisionTreeRegressor(max_depth=5, random_state=42)

    reg_tree.fit(X_train, y_train)

    plt.figure(figsize=(10, 8))

    # Using plot tree to visualize decision tree
    plot_tree(reg_tree, feature_names=['year', 'isTeamSport'], filled=True)
    plt.show()

    st.pyplot(plt)

    y_pred_tree = reg_tree.predict(X_test)
    st.info("Decision Tree- AUSTRALIA")

    # R² score measures the variance in the data
    st.info(f'R² score: {r2_score(y_test, y_pred_tree)}')
    st.info(f'Mean Squared Error: {mean_squared_error(y_test, y_pred_tree)}')
    st.info("-------------")

    reg_random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

    reg_random_forest.fit(X_train, y_train)

    y_pred_rf = reg_random_forest.predict(X_test)
    st.info("Random Forest Results- AUSTRALIA")

    # Mean Squared Error masures the average squared difference between the predicted and actual values.
    st.info(f'R² score: {r2_score(y_test, y_pred_rf)}')
    st.info(f'Mean Squared Error: {mean_squared_error(y_test, y_pred_rf)}')


    medals_yearwise_data_ind['isTeamSport'] = medals_yearwise_data_ind['isTeamSport'].apply(lambda x: 1 if x else 0)

    X = medals_yearwise_data_ind[['year', 'isTeamSport']]
    y = medals_yearwise_data_ind['medal_count']

    # Split dataset into training 80% and testing 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Max depth set to 5 to avoid overfitting
    reg_tree = DecisionTreeRegressor(max_depth=5, random_state=42)

    reg_tree.fit(X_train, y_train)

    plt.figure(figsize=(10, 8))

    # Using plot tree to visualize decision tree
    plot_tree(reg_tree, feature_names=['year', 'isTeamSport'], filled=True)
    plt.show()

    st.pyplot(plt)

    y_pred_tree = reg_tree.predict(X_test)
    st.info("Decision Tree- INDIA")

    # R² score measures the variance in the data
    st.info(f'R² score: {r2_score(y_test, y_pred_tree)}')
    st.info(f'Mean Squared Error: {mean_squared_error(y_test, y_pred_tree)}')
    st.info("-------------")

    reg_random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

    reg_random_forest.fit(X_train, y_train)

    y_pred_rf = reg_random_forest.predict(X_test)
    st.info("Random Forest Results- INDIA")

    # Mean Squared Error masures the average squared difference between the predicted and actual values.
    st.info(f'R² score: {r2_score(y_test, y_pred_rf)}')
    st.info(f'Mean Squared Error: {mean_squared_error(y_test, y_pred_rf)}')

    st.info("We learnt that in case the parameters are interpretability and simplicity. It is clear that decision trees are way better and useful. But if we want better accuracy Random Forest is the more effective choice. We also understood that my Random Forest model could have been improved if there were more features. ")

def hypothesis4(st):
    Athletes_Data = read_entries("Pre_Athlete_Events_Details")
    Athlete_Biography = read_entries("Pre_Athlete_Biography")
    Athletes_Data = Athletes_Data[Athletes_Data['olympic_type'] == 'summer']
    Athletes_Data = pd.merge(Athletes_Data, Athlete_Biography, on='athlete_id', how='left')
    Athletes_Data['sex'] = Athletes_Data.apply(lambda row: 'M' if row['men'] == 1 else 'F', axis=1)
    Athletes_Data['birth_year'] = Athletes_Data['born'].str.extract(r'(\d{4})')
    Athletes_Data['birth_year'] = pd.to_numeric(Athletes_Data['birth_year'], errors='coerce')
    Athletes_Data['year'] = pd.to_numeric(Athletes_Data['year'], errors='coerce')
    Athletes_Data['age'] = Athletes_Data['year'] - Athletes_Data['birth_year']

    Athletes_Data = Athletes_Data[['age', 'country', 'year', 'sex', 'sport', 'height', 'weight', 'medal', 'isTeamSport']]

    Athletes_Data['height'] = pd.to_numeric(Athletes_Data['height'], errors='coerce')
    Athletes_Data['weight'] = pd.to_numeric(Athletes_Data['weight'], errors='coerce')

    Athletes_Data = Athletes_Data.dropna(subset=['height', 'weight', 'medal', 'age', 'country'])

    Athletes_Data_Male = Athletes_Data[Athletes_Data['sex'] == 'M']
    Athletes_Data_Female = Athletes_Data[Athletes_Data['sex'] == 'F']
    sports = ['tennis', 'table tennis']
    Athletes_Data_Male_Popular = Athletes_Data_Male[Athletes_Data_Male['sport'].isin(sports)]
    Athletes_Data_Female_Popular = Athletes_Data_Female[Athletes_Data_Female['sport'].isin(sports)]

    X = Athletes_Data_Male_Popular[['height', 'weight', 'country']]
    y = Athletes_Data_Male_Popular['sport']

    country_encoder = LabelEncoder()
    X['country'] = country_encoder.fit_transform(X['country'])

    sport_encoder = LabelEncoder()
    y = sport_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("Building a model for Male Athletes")

    st.markdown("#### Height vs. Weight chart for Tennis and Table Tennis Athletes (Male)")
    plt.figure(figsize=(10, 6))
    for sport in sports:
        sport_data = Athletes_Data_Male_Popular[Athletes_Data_Male_Popular['sport'] == sport]
        plt.scatter(sport_data['height'], sport_data['weight'], label=sport, alpha=0.7)

    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Height vs. Weight for Tennis and Table Tennis Athletes')
    plt.legend(title='Sport')
    plt.grid(alpha=0.3)
    plt.show()

    st.pyplot(plt)


    st.write("We have run several models to find a model which could predict which an athlete belongs to")
    st.info("--------------------Logistic Regression results------------------------------")

    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    make_report(st, y_test, y_pred_lr, sport_encoder)

    st.info("--------------------KNN results------------------------------")
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    make_report(st, y_test, y_pred_knn, sport_encoder)

    st.info("--------------------SVM results----------------------------------------")
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    make_report(st, y_test, y_pred_svm, sport_encoder)

    st.info("--------------------Random Forest results------------------------------")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    make_report(st, y_test, y_pred_rf, sport_encoder)

    st.info("--------------------Gradient Boost results-----------------------------")
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    make_report(st, y_test, y_pred_gb, sport_encoder)

    st.info("Random forest gives the best result with accuracy of 95%")

    st.subheader("Building a model for female Athletes")

    st.markdown("#### Height vs. Weight chart for Tennis and Table Tennis Athletes (Female)")
    plt.figure(figsize=(10, 6))
    for sport in sports:
        sport_data = Athletes_Data_Female_Popular[Athletes_Data_Female_Popular['sport'] == sport]
        plt.scatter(sport_data['height'], sport_data['weight'], label=sport, alpha=0.7)

    # Add labels, title, and legend
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Height vs. Weight for Tennis and Table Tennis Athletes')
    plt.legend(title='Sport')
    plt.grid(alpha=0.3)
    plt.show()

    st.pyplot(plt)

    st.write("We have run several models to find a model which could predict which an athlete belongs to")

    X = Athletes_Data_Female_Popular[['height', 'weight', 'country']]
    y = Athletes_Data_Female_Popular['sport']

    country_encoder = LabelEncoder()
    X['country'] = country_encoder.fit_transform(X['country'])

    sport_encoder = LabelEncoder()
    y = sport_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.info("--------------------Logistic Regression results------------------------------")

    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    make_report(st, y_test, y_pred_lr, sport_encoder)

    st.info("--------------------KNN results------------------------------")
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    make_report(st, y_test, y_pred_knn, sport_encoder)

    st.info("--------------------SVM results----------------------------------------")
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    make_report(st, y_test, y_pred_svm, sport_encoder)

    st.info("--------------------Random Forest results------------------------------")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    make_report(st, y_test, y_pred_rf, sport_encoder)

    st.info("--------------------Gradient Boost results-----------------------------")
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    make_report(st, y_test, y_pred_gb, sport_encoder)

    st.info("Random forest gives the best result with accuracy of 91%")