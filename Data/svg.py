import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from matplotlib.table import Table

# Create the data without R²
data = {
    'Metric': ['Cost Reduction (%)', 'MAPE', 'RMSE', 'QoS Violation (%)', 'Response Time (ms)'],
    'Experiment Value': [95.39, 3.87, 227.52, 1.5, 2985],
    'Expected Value': ['95.39', '≤ 5.61', '≤ 270.47', '≤ 5.00', '≤ 5000'],
    'Status': ['✓', '✓', '✓', '✓', '✓']
}

# Create DataFrame
df = pd.DataFrame(data)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
ax.axis('tight')

# Create table
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center',
    colColours=['#f2f2f2']*4,
    colWidths=[0.25, 0.25, 0.25, 0.15]
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

# Style header row
for k, cell in table.get_celld().items():
    if k[0] == 0:  # Header row
        cell.set_text_props(weight='bold', color='black')
        cell.set_facecolor('#d9d9d9')
    elif k[1] == 3:  # Status column
        if cell.get_text().get_text() == '✓':
            cell.set_facecolor('#e6ffe6')  # Light green for pass
        else:
            cell.set_facecolor('#ffe6e6')  # Light red for fail

# Add title
plt.title('Validation Results vs Paper Benchmarks', fontsize=16, pad=20)

# Save as SVG to a string
buf = io.BytesIO()
plt.savefig(buf, format='svg', bbox_inches='tight')
plt.close(fig)

# Get the SVG code
svg_code = buf.getvalue().decode('utf-8')
buf.close()

# Print the SVG code
# Save the SVG code into a file
svg_filename = 'validation_results.svg'
with open(svg_filename, 'w') as file:
    file.write(svg_code)

print(f"SVG saved to {svg_filename}")

