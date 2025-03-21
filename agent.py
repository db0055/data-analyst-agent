import os
import re
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq

class DataAnalystAgent:
    def __init__(self, data, file_type):
        """
        Initialize the data analyst agent.
        
        Args:
            data: The processed data (DataFrame or text)
            file_type: Type of the data ("tabular", "text", "pdf", "docx", "image")
        """
        self.data = data
        self.file_type = file_type
        
        # Get API key from environment variable or use a default for development
        self.api_key = os.getenv("GROQ_API_KEY")
        
        # If no API key is found, use a default value (for development only)
        if not self.api_key:
            # Replace this with your actual Groq API key
            self.api_key = "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"  # Replace with your Groq API key
        
        # Initialize the Groq client with just the API key, no additional parameters
        self.client = Groq(api_key=self.api_key)
        # Use a smaller model to avoid token limits
        self.model = "llama3-8b-8192"
    
    
    def process_query(self, query):
        """
        Process a user query and return the response.
        
        Args:
            query: The user's question or request
        
        Returns:
            Either a string response or a tuple (fig, explanation, chart_code) for visualizations
        """
        # Check if data is valid
        if self.data is None:
            return "No data has been uploaded. Please upload a file first."
            
        # Simple validation for DataFrame (tabular data)
        if self.file_type == "tabular":
            if not isinstance(self.data, pd.DataFrame):
                return "The uploaded file doesn't contain valid tabular data."
            if self.data.empty:
                return "The uploaded file contains an empty dataset. Please upload a file with data."
                
        # Check visualization keywords or analysis commands
        visualization_keywords = [
            "plot", "chart", "graph", "visualize", "show", "display", 
            "histogram", "bar chart", "scatter plot", "line chart", "pie chart",
            "visualization", "analyse", "analyze", "analysis", "summarize", "summarise",
            "summary", "dashboard", "metrics", "statistics", "stats", "distribution",
            "correlation", "trend", "patterns", "insight", "overview"
        ]
        
        # Handle different file types
        if self.file_type == "tabular":
            # Process tabular data (CSV, Excel)
            if any(keyword in query.lower() for keyword in visualization_keywords):
                # Check for specific visualization requests
                if "pie chart" in query.lower() or "pie" in query.lower():
                    return self._create_pie_chart(query)
                elif "bar chart" in query.lower() or "bar graph" in query.lower():
                    return self._create_bar_chart(query)
                elif "line chart" in query.lower() or "line graph" in query.lower() or "trend" in query.lower():
                    return self._create_line_chart(query)
                elif "scatter" in query.lower() or "correlation" in query.lower():
                    return self._create_scatter_plot(query)
                elif "histogram" in query.lower() or "distribution" in query.lower():
                    return self._create_histogram(query)
                elif "summarize" in query.lower() or "summary" in query.lower():
                    return self._generate_data_summary()
                else:
                    # Generic visualization or analysis request
                    return self._process_visualization_query(query)
            else:
                return self._process_tabular_query(query)
        
        elif self.file_type in ["text", "pdf", "docx", "image"]:
            return self._process_text_query(query)
        
        else:
            return "I'm not sure how to process this type of data. Currently supported file types include CSV, Excel, PDF, Word, text, and image files."
    
    def _generate_data_summary(self):
        """Generate a comprehensive summary of the data."""
        try:
            # Basic statistics
            summary_parts = []
            
            # Basic info
            summary_parts.append(f"Dataset shape: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            summary_parts.append(f"Columns: {', '.join(self.data.columns.tolist())}")
            
            # Missing values
            missing = self.data.isnull().sum()
            if missing.sum() > 0:
                missing_cols = missing[missing > 0]
                summary_parts.append(f"Missing values found in {len(missing_cols)} columns")
                for col, count in missing_cols.items():
                    pct = (count / len(self.data)) * 100
                    summary_parts.append(f"  - {col}: {count} values ({pct:.1f}%)")
            else:
                summary_parts.append("No missing values found in the dataset")
            
            # Numerical columns analysis
            num_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
            if not num_cols.empty:
                summary_parts.append("\nNumerical Columns Analysis:")
                desc = self.data[num_cols].describe().round(2)
                stats_text = []
                for col in num_cols:
                    stats = desc[col]
                    stats_text.append(f"  - {col}: Range [{stats['min']} to {stats['max']}], Mean: {stats['mean']}, Median: {stats['50%']}")
                summary_parts.append("\n".join(stats_text))
            
            # Categorical columns analysis
            cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
            if not cat_cols.empty:
                summary_parts.append("\nCategorical Columns Analysis:")
                for col in cat_cols[:5]:  # Limit to first 5 categorical columns
                    value_counts = self.data[col].value_counts()
                    top_values = value_counts.head(3)
                    summary_parts.append(f"  - {col}: {self.data[col].nunique()} unique values")
                    summary_parts.append(f"    Top values: " + ", ".join([f"{v} ({c})" for v, c in top_values.items()]))
            
            # Create a summary visualization
            try:
                # Choose the most appropriate visualization based on data
                if len(num_cols) >= 2:
                    # Create correlation heatmap
                    corr = self.data[num_cols].corr().round(2)
                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        title='Correlation Matrix of Numerical Variables'
                    )
                    explanation = "Here's a comprehensive summary of your dataset with a correlation heatmap of numerical variables."
                    return fig, "\n".join(summary_parts) + "\n\n" + explanation
                
                elif len(num_cols) == 1:
                    # Create histogram for the single numerical column
                    fig = px.histogram(
                        self.data,
                        x=num_cols[0],
                        title=f'Distribution of {num_cols[0]}'
                    )
                    explanation = f"Here's a comprehensive summary of your dataset with a histogram showing the distribution of {num_cols[0]}."
                    return fig, "\n".join(summary_parts) + "\n\n" + explanation
                
                elif len(cat_cols) > 0:
                    # Create bar chart for the first categorical column
                    counts = self.data[cat_cols[0]].value_counts().head(10).reset_index()
                    counts.columns = [cat_cols[0], 'count']
                    fig = px.bar(
                        counts,
                        x=cat_cols[0],
                        y='count',
                        title=f'Top 10 values in {cat_cols[0]}'
                    )
                    explanation = f"Here's a comprehensive summary of your dataset with a bar chart showing the most frequent values in {cat_cols[0]}."
                    return fig, "\n".join(summary_parts) + "\n\n" + explanation
            
            except Exception as vis_error:
                # Return just the text summary if visualization fails
                return "\n".join(summary_parts) + f"\n\nNote: Unable to create visualization ({str(vis_error)})"
            
            # If no visualization created, return just the text
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error generating data summary: {str(e)}"
    
    def _create_pie_chart(self, query):
        """Create a pie chart based on the query."""
        try:
            # Identify which column to use for the pie chart
            cat_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            if not cat_cols:
                return "No categorical columns found for creating a pie chart."
            
            # Use the first categorical column by default
            target_col = cat_cols[0]
            
            # Try to find a relevant column based on the query
            for col in cat_cols:
                if col.lower() in query.lower():
                    target_col = col
                    break
            
            # Get value counts for the column
            value_counts = self.data[target_col].value_counts()
            
            # Limit to top 10 categories for readability
            if len(value_counts) > 10:
                top_values = value_counts.head(9)
                other_sum = value_counts[9:].sum()
                top_values['Other'] = other_sum
                value_counts = top_values
            
            # Create pie chart
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'Distribution of {target_col}'
            )
            
            # Add percentage labels
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            explanation = f"Here's a pie chart showing the distribution of {target_col}."
            return fig, explanation
            
        except Exception as e:
            return f"Error creating pie chart: {str(e)}"
    
    def _create_bar_chart(self, query):
        """Create a bar chart based on the query."""
        try:
            # Try to identify which columns to use
            cat_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            num_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if not cat_cols:
                # If no categorical columns, create a bar chart of numerical column counts
                if not num_cols:
                    return "No suitable columns found for creating a bar chart."
                    
                # Use the first numerical column
                fig = px.histogram(
                    self.data,
                    x=num_cols[0],
                    title=f'Distribution of {num_cols[0]}'
                )
                explanation = f"Here's a bar chart showing the distribution of {num_cols[0]}."
                return fig, explanation
            
            # Use first categorical and numerical columns by default
            x_col = cat_cols[0]
            y_col = num_cols[0] if num_cols else None
            
            # Try to find relevant columns based on the query
            for col in cat_cols:
                if col.lower() in query.lower():
                    x_col = col
                    break
            
            if y_col:
                for col in num_cols:
                    if col.lower() in query.lower():
                        y_col = col
                        break
                
                # Group by categorical column and calculate mean of numerical column
                grouped_data = self.data.groupby(x_col)[y_col].mean().reset_index()
                
                # Sort and limit to top 15 for readability
                if len(grouped_data) > 15:
                    grouped_data = grouped_data.sort_values(y_col, ascending=False).head(15)
                
                # Create bar chart
                fig = px.bar(
                    grouped_data,
                    x=x_col,
                    y=y_col,
                    title=f'Average {y_col} by {x_col}'
                )
                explanation = f"Here's a bar chart showing the average {y_col} for each {x_col}."
            else:
                # Just count occurrences of categorical column
                counts = self.data[x_col].value_counts().reset_index()
                counts.columns = [x_col, 'count']
                
                # Limit to top 15 for readability
                if len(counts) > 15:
                    counts = counts.head(15)
                
                # Create bar chart
                fig = px.bar(
                    counts,
                    x=x_col,
                    y='count',
                    title=f'Count of {x_col}'
                )
                explanation = f"Here's a bar chart showing the count of each {x_col}."
            
            return fig, explanation
            
        except Exception as e:
            return f"Error creating bar chart: {str(e)}"
    
    def _create_line_chart(self, query):
        """Create a line chart based on the query."""
        try:
            # Check for date/time columns
            date_cols = []
            for col in self.data.columns:
                # Try to convert to datetime
                try:
                    pd.to_datetime(self.data[col])
                    date_cols.append(col)
                except:
                    continue
            
            # Try to identify which columns to use
            num_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if not date_cols and not num_cols:
                return "No suitable columns found for creating a line chart."
            
            # Use first date and numerical columns by default
            x_col = date_cols[0] if date_cols else num_cols[0]
            y_col = num_cols[0] if num_cols and x_col != num_cols[0] else (num_cols[1] if len(num_cols) > 1 else None)
            
            if not y_col:
                return "Not enough numerical columns for creating a meaningful line chart."
            
            # Try to find relevant columns based on the query
            for col in self.data.columns:
                if col.lower() in query.lower():
                    if col in date_cols or col in num_cols:
                        x_col = col
                        # Reset y_col if it's the same as new x_col
                        if y_col == x_col:
                            y_col = next((c for c in num_cols if c != x_col), None)
                        break
            
            for col in num_cols:
                if col.lower() in query.lower() and col != x_col:
                    y_col = col
                    break
            
            # Prepare data
            plot_data = self.data[[x_col, y_col]].dropna()
            
            # Sort by x_col if it's a date
            if x_col in date_cols:
                plot_data[x_col] = pd.to_datetime(plot_data[x_col])
                plot_data = plot_data.sort_values(x_col)
            
            # Create line chart
            fig = px.line(
                plot_data,
                x=x_col,
                y=y_col,
                title=f'{y_col} over {x_col}'
            )
            
            explanation = f"Here's a line chart showing {y_col} plotted against {x_col}."
            return fig, explanation
            
        except Exception as e:
            return f"Error creating line chart: {str(e)}"
    
    def _create_scatter_plot(self, query):
        """Create a scatter plot based on the query."""
        try:
            # Try to identify which columns to use
            num_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(num_cols) < 2:
                return "Not enough numerical columns for creating a scatter plot."
            
            # Use first two numerical columns by default
            x_col = num_cols[0]
            y_col = num_cols[1]
            
            # Try to find relevant columns based on the query
            for col in num_cols:
                if col.lower() in query.lower():
                    if col in num_cols:
                        x_col = col
                        # Reset y_col if it's the same as new x_col
                        if y_col == x_col:
                            y_col = next((c for c in num_cols if c != x_col), None)
                    break
            
            for col in num_cols:
                if col.lower() in query.lower() and col != x_col:
                    y_col = col
                    break
            
            # Prepare data
            plot_data = self.data[[x_col, y_col]].dropna()
            
            # Add third numerical column as bubble size if available
            color_col = None
            if len(num_cols) > 2:
                color_options = [col for col in num_cols if col != x_col and col != y_col]
                if color_options:
                    color_col = color_options[0]
                    # Try to find a relevant third column
                    for col in color_options:
                        if col.lower() in query.lower():
                            color_col = col
                            break
            
            # Create scatter plot
            if color_col:
                fig = px.scatter(
                    plot_data,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f'{y_col} vs {x_col} (colored by {color_col})'
                )
                explanation = f"Here's a scatter plot showing the relationship between {x_col} and {y_col}, with points colored by {color_col}."
            else:
                fig = px.scatter(
                    plot_data,
                    x=x_col,
                    y=y_col,
                    title=f'{y_col} vs {x_col}'
                )
                explanation = f"Here's a scatter plot showing the relationship between {x_col} and {y_col}."
            
            # Add trendline
            fig.update_layout(
                shapes=[{
                    'type': 'line',
                    'x0': plot_data[x_col].min(),
                    'y0': plot_data[y_col].min(),
                    'x1': plot_data[x_col].max(),
                    'y1': plot_data[y_col].max(),
                    'line': {
                        'color': 'red',
                        'width': 1,
                        'dash': 'dash',
                    }
                }]
            )
            
            return fig, explanation
            
        except Exception as e:
            return f"Error creating scatter plot: {str(e)}"
    
    def _create_histogram(self, query):
        """Create a histogram based on the query."""
        try:
            # Try to identify which column to use
            num_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if not num_cols:
                return "No numerical columns found for creating a histogram."
            
            # Use first numerical column by default
            target_col = num_cols[0]
            
            # Try to find a relevant column based on the query
            for col in num_cols:
                if col.lower() in query.lower():
                    target_col = col
                    break
            
            # Create histogram
            fig = px.histogram(
                self.data,
                x=target_col,
                nbins=20,
                title=f'Distribution of {target_col}'
            )
            
            # Add a curve of the distribution
            fig.update_layout(
                xaxis_title=target_col,
                yaxis_title="Count"
            )
            
            explanation = f"Here's a histogram showing the distribution of {target_col}."
            return fig, explanation
            
        except Exception as e:
            return f"Error creating histogram: {str(e)}"
    
    def _process_tabular_query(self, query):
        """Process queries for tabular data without visualization."""
        # Get data info
        data_info = self._get_detailed_data_info()
        
        # Construct the prompt
        prompt = f"""
        I have a DataFrame with the following information:
        {data_info}
        
        User question: {query}
        
        Please answer the question based on the data information provided.
        Give a detailed and informative answer.
        """
        
        # Get response from Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying AI: {str(e)}"
    
    def _process_visualization_query(self, query):
        """Process queries for tabular data that require visualization."""
        # Get data info
        data_info = self._get_detailed_data_info()
        
        # Limit data sample for large datasets
        data_sample = self.data
        if len(self.data) > 1000:
            data_sample = self.data.sample(1000, random_state=42)
        
        # Use a sample for preview
        sample_rows = min(5, len(data_sample))
        
        # Construct the prompt
        prompt = f"""
        I have a DataFrame with the following information:
        {data_info}
        
        The first few rows of data (sample):
        {data_sample.head(sample_rows).to_string()}
        
        User request: {query}
        
        Please create a visualization for this request using Plotly Express or Plotly Graph Objects.
        Focus ONLY on the visualization code - be concise.
        
        Format your response as:
        ```explanation
        Brief explanation - 2-3 sentences only
        ```
        
        ```python
        # Visualization code
        ```
        """
        
        # Get response from Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            # Extract explanation and code
            content = response.choices[0].message.content
            explanation = self._extract_explanation(content)
            chart_code = self._extract_code(content)
            
            if chart_code:
                # Execute the code to generate the figure
                local_vars = {"df": data_sample, "px": px, "go": go, "pd": pd, "np": np}
                try:
                    exec(chart_code, globals(), local_vars)
                    if "fig" in local_vars:
                        return local_vars["fig"], explanation, chart_code
                except Exception as e:
                    # Try to create a default visualization if custom one fails
                    try:
                        # Create a default chart based on data types
                        if not data_sample.empty:
                            num_cols = data_sample.select_dtypes(include=['int64', 'float64']).columns
                            cat_cols = data_sample.select_dtypes(include=['object', 'category']).columns
                            
                            if len(num_cols) >= 1 and len(cat_cols) >= 1:
                                # Create a bar chart using the first categorical and numerical columns
                                fig = px.bar(
                                    data_sample, 
                                    x=cat_cols[0], 
                                    y=num_cols[0], 
                                    title=f"{num_cols[0]} by {cat_cols[0]}",
                                    labels={cat_cols[0]: cat_cols[0], num_cols[0]: num_cols[0]}
                                )
                                explanation = f"Here's a bar chart showing {num_cols[0]} by {cat_cols[0]}. The original visualization failed with error: {str(e)}"
                                return fig, explanation, "# Default visualization created"
                            elif len(num_cols) >= 2:
                                # Create a scatter plot using the first two numerical columns
                                fig = px.scatter(
                                    data_sample, 
                                    x=num_cols[0], 
                                    y=num_cols[1], 
                                    title=f"{num_cols[1]} vs {num_cols[0]}",
                                    labels={num_cols[0]: num_cols[0], num_cols[1]: num_cols[1]}
                                )
                                explanation = f"Here's a scatter plot showing {num_cols[1]} vs {num_cols[0]}. The original visualization failed with error: {str(e)}"
                                return fig, explanation, "# Default visualization created"
                            else:
                                # Create a simple count plot for the first categorical column
                                counts = data_sample[cat_cols[0]].value_counts().reset_index()
                                counts.columns = [cat_cols[0], 'count']
                                fig = px.bar(
                                    counts, 
                                    x=cat_cols[0], 
                                    y='count', 
                                    title=f"Count of {cat_cols[0]}",
                                    labels={cat_cols[0]: cat_cols[0], 'count': 'Count'}
                                )
                                explanation = f"Here's a count plot for {cat_cols[0]}. The original visualization failed with error: {str(e)}"
                                return fig, explanation, "# Default visualization created"
                    except Exception as fallback_error:
                        pass
                        
                    return f"Error generating visualization: {str(e)}\n\nCode:\n{chart_code}", None, None
            
            return explanation, None, None
        except Exception as e:
            return f"Error querying AI: {str(e)}", None, None
    
    def _process_text_query(self, query):
        """Process queries for text data."""
        # Use full text instead of truncating
        text_data = self.data
        
        # Construct the prompt
        prompt = f"""
        I have the following text data:
        
        {text_data}
        
        User question: {query}
        
        Please analyze this text and answer the question. 
        Give a detailed and informative answer.
        """
        
        # Get response from Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying AI: {str(e)}"
    
    def _get_limited_data_info(self):
        """Get limited information about the DataFrame to avoid token limit issues."""
        if self.file_type != "tabular":
            return "Not tabular data"
        
        info = []
        
        # Basic DataFrame info
        info.append(f"DataFrame Shape: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Column names only (no types to save tokens)
        info.append("Columns: " + ", ".join(self.data.columns.tolist()))
        
        # Get a very limited summary of data types
        num_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if num_cols:
            info.append(f"Numerical columns: {', '.join(num_cols[:5])}" + ("..." if len(num_cols) > 5 else ""))
        
        if cat_cols:
            info.append(f"Categorical columns: {', '.join(cat_cols[:5])}" + ("..." if len(cat_cols) > 5 else ""))
        
        return "\n".join(info)
    
    def _get_detailed_data_info(self):
        """Get more detailed information about the DataFrame."""
        if self.file_type != "tabular":
            return "Not tabular data"
        
        info = []
        
        # Basic DataFrame info
        info.append(f"DataFrame Shape: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Column names and types
        cols_info = []
        for col in self.data.columns:
            dtype = self.data[col].dtype
            nunique = self.data[col].nunique()
            missing = self.data[col].isnull().sum()
            missing_pct = (missing / len(self.data)) * 100 if len(self.data) > 0 else 0
            
            col_info = f"{col} (type: {dtype}, unique values: {nunique}"
            if missing > 0:
                col_info += f", missing: {missing} ({missing_pct:.1f}%)"
            col_info += ")"
            
            cols_info.append(col_info)
        
        info.append("Columns:\n- " + "\n- ".join(cols_info))
        
        # Sample data statistics
        num_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if num_cols:
            # Get basic stats for numerical columns
            stats = self.data[num_cols].describe().round(2).to_string()
            info.append(f"Numerical Statistics:\n{stats}")
        
        # Sample of categorical data
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            cat_info = []
            for col in cat_cols[:3]:  # Limit to first 3 categorical columns
                values = self.data[col].value_counts().head(5).to_dict()
                values_str = ", ".join([f"{k}: {v}" for k, v in values.items()])
                cat_info.append(f"{col} - Top values: {values_str}")
            
            if cat_info:
                info.append("Categorical Data Sample:\n- " + "\n- ".join(cat_info))
        
        return "\n".join(info)
    
    def _extract_code(self, text):
        """Extract Python code from the text."""
        code_pattern = re.compile(r'```(?:python)?\s*(.*?)\s*```', re.DOTALL)
        match = code_pattern.search(text)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_explanation(self, text):
        """Extract explanation from the text."""
        explanation_pattern = re.compile(r'```explanation\s*(.*?)\s*```', re.DOTALL)
        match = explanation_pattern.search(text)
        if match:
            return match.group(1).strip()
        
        # If no explanation block is found, try to extract the text before code block
        code_index = text.find("```python")
        if code_index == -1:
            code_index = text.find("```")
        
        if code_index > 0:
            return text[:code_index].strip()
        
        return text.strip()