import streamlit as st
 
st.title('Departments')
 
# Create tabs
tab_titles = ['HR Support', 'IT Support', 'Transportation Support', 'Other']
tabs = st.tabs(tab_titles)
 
# Add content to each tab...
with tabs[0]:
    st.header('HR Support tickets')
    for ticket in st.session_state.categories["HR"]:
        st.write(str( st.session_state.categories["HR"].index(ticket)+1)+" : "+ticket)
    
with tabs[1]:
    st.header('IT Support tickets')
    for ticket in st.session_state.categories['IT']:
        st.write(str(st.session_state.categories['IT'].index(ticket)+1)+" : "+ticket)
 
with tabs[2]:
    st.header('Transportation Support tickets')
    for ticket in st.session_state.categories['Transportation']:
        st.write(str(st.session_state.categories['Transportation'].index(ticket)+1)+" : "+ticket)
    
with tabs[3]:
    st.header('Other tickets')
    for ticket in st.session_state.categories['Other']:
        st.write(str(st.session_state.categories['Other'].index(ticket)+1)+" : "+ticket)

