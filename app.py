import streamlit as st
import pandas as pd
import pickle

# Title
st.write('''
        # Customer Cost Prediction
''')

st.write('---')

# Model path
MODEL_PATH = 'model.pkl'

# DataFrame
customer = pd.read_csv('clean_data.csv')
customer = customer.drop(
    columns=['index', 'cost', 'avg_cars_at_home1', 'store_sqft', 'grocery_sqft', 'frozen_sqft', 'meat_sqft'], axis=1)
st.subheader('Data')
st.write(customer)


def user_input():
    food_category = st.selectbox('Food Category', customer.food_category.value_counts().index)
    food_department = st.selectbox('Food Department', customer.food_department.value_counts().index)
    food_family = st.selectbox('Food Family', customer.food_family.value_counts().index)
    store_sales = st.slider('Store Sales', 0.51, 22.92, 6.54)
    store_cost = st.slider('Store Cost', 0.16, 9.72, 2.61)
    unit_sales = st.slider('Unit Sales', 1.00, 6.00, 3.09)
    promotion_name = st.selectbox('Promotion Name', customer.promotion_name.value_counts().index)
    sales_country = st.selectbox('Sales Country', customer.sales_country.value_counts().index)
    marital_status = st.selectbox('Marital Status', customer.marital_status.value_counts().index)
    gender = st.selectbox('Gender', customer.gender.value_counts().index)
    total_children = st.selectbox('Total Children', customer.total_children.value_counts().index)
    education = st.selectbox('Education', customer.education.value_counts().index)
    member_card = st.selectbox('Member Card', customer.member_card.value_counts().index)
    occupation = st.selectbox('Occupation', customer.occupation.value_counts().index)
    houseowner = st.selectbox('House Owner', customer.houseowner.value_counts().index)
    avg_cars_at_home = st.selectbox('Average cars at home', customer.avg_cars_at_home.value_counts().index)
    avg_yearly_income = st.selectbox('Average yearly income', customer.avg_yearly_income.value_counts().index)
    num_children_at_home = st.selectbox('Num children at home', customer.num_children_at_home.value_counts().index)
    brand_name = st.selectbox('Brand Name', customer.brand_name.value_counts().index)
    SRP = st.slider('SRP', 0.50, 3.98, 2.11)
    gross_weight = st.slider('Gross Weight', 6.00, 21.90, 13.80)
    net_weight = st.slider('Net Weight', 3.05, 20.80, 11.79)
    recyclable_package = st.selectbox('Recyclable Package', customer.recyclable_package.value_counts().index)
    low_fat = st.selectbox('Low Fat', customer.low_fat.value_counts().index)
    units_per_case = st.slider('Units per case', 1, 36, 18)
    store_type = st.selectbox('Store Type', customer.store_type.value_counts().index)
    store_city = st.selectbox('Store City', customer.store_city.value_counts().index)
    store_state = st.selectbox('Store State', customer.store_state.value_counts().index)
    coffee_bar = st.selectbox('Coffee Bar', customer.coffee_bar.value_counts().index)
    video_store = st.selectbox('Video Store', customer.video_store.value_counts().index)
    salad_bar = st.selectbox('Salad Bar', customer.salad_bar.value_counts().index)
    prepared_food = st.selectbox('Prepared Food', customer.prepared_food.value_counts().index)
    florist = st.selectbox('Florist', customer.florist.value_counts().index)
    media_type = st.selectbox('Media Type', customer.media_type.value_counts().index)


    # Create dictionary with all inputs
    data_row = {
        'food_category': food_category,
        'food_department': food_department,
        'food_family': food_family,
        'store_sales': store_sales,
        'store_cost': store_cost,
        'unit_sales': unit_sales,
        'promotion_name': promotion_name,
        'sales_country': sales_country,
        'marital_status': marital_status,
        'gender': gender,
        'total_children': total_children,
        'education': education,
        'member_card': member_card,
        'occupation': occupation,
        'houseowner': houseowner,
        'avg_cars_at_home': avg_cars_at_home,
        'avg_yearly_income': avg_yearly_income,
        'num_children_at_home': num_children_at_home,
        'brand_name': brand_name,
        'SRP': SRP,
        'gross_weight': gross_weight,
        'net_weight':net_weight,
        'recyclable_package':recyclable_package,
        'low_fat':low_fat,
        'units_per_case':units_per_case,
        'store_type':store_type,
        'store_city':store_city,
        'store_state':store_state,
        'coffee_bar':coffee_bar,
        'video_store':video_store,
        'salad_bar':salad_bar,
        'prepared_food':prepared_food,
        'florist':florist,
        'media_type':media_type
    }

    # Create info dataframe
    info = pd.DataFrame(data_row, index=[0])
    return info


info = user_input()


st.subheader('Model prediction')

# Drop unnecessary columns
df = pd.concat([info, customer], axis=0)
st.write(df[:1])    # Show the first line

if st.button('Predict'):
    # Load model
    model=''
    if model=='':
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)

    # Prediction
    pred = model.predict(df)
    st.write('$',pred[0])
    st.write('---')
