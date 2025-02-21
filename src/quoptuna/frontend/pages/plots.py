import streamlit as st
import optuna
import plotly.graph_objects as go
from quoptuna.frontend.support import handle_input, initialize_session_state

def app():
    st.title("Optimization Plots")
    
    # Initialize session state and get optimizer
    initialize_session_state()
    optimizer, study_name, db_name = handle_input(context="plots")
    
    if optimizer and study_name:
        loaded_study = optuna.load_study(study_name=study_name, storage=optimizer.storage_location)
        
        # Create tabs for different plot types
        plot_type = st.selectbox(
            "Select Plot Type",
            [
                "Optimization History",
                "Parameter Importances",
                "Parameter Relationships",
                "Timeline",
                "Intermediate Values",
                "Parallel Coordinate"
            ],
            key="plot_type_select"  # Add unique key for selectbox
        )
        
        try:
            if plot_type == "Optimization History":
                fig = optuna.visualization.plot_optimization_history(loaded_study)
                st.plotly_chart(fig)
                
            elif plot_type == "Parameter Importances":
                fig = optuna.visualization.plot_param_importances(loaded_study)
                st.plotly_chart(fig)
                
            elif plot_type == "Parameter Relationships":
                fig = optuna.visualization.plot_parallel_coordinate(loaded_study)
                st.plotly_chart(fig)
                
            elif plot_type == "Timeline":
                fig = optuna.visualization.plot_timeline(loaded_study)
                st.plotly_chart(fig)
                
            elif plot_type == "Intermediate Values":
                fig = optuna.visualization.plot_intermediate_values(loaded_study)
                st.plotly_chart(fig)
                
            elif plot_type == "Parallel Coordinate":
                fig = optuna.visualization.plot_parallel_coordinate(loaded_study)
                st.plotly_chart(fig)
            
            # Add download button for the plot
            if st.button("Download Plot", key="download_plot_button_plots"):  # Add unique key
                # Convert plot to HTML
                plot_html = fig.to_html()
                st.download_button(
                    label="Download Plot as HTML",
                    data=plot_html,
                    file_name=f"quoptuna_{plot_type.lower().replace(' ', '_')}.html",
                    mime="text/html",
                    key="download_html_button_plots"  # Add unique key
                )
                
        except (ValueError, AttributeError) as e:
            st.error(f"Error generating plot: {str(e)}")
            st.info("This error might occur if there isn't enough data for the selected plot type.")
    else:
        st.warning("Please load an optimizer and study first using the sidebar.")

if __name__ == "__main__":
    app() 