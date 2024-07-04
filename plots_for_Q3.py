import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file with the specified encoding
file_path = "C:/Shay/UNI/yearB/IML/HACKTHON/train_bus_schedule.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-8')

# Convert the 'arrival_time' column to datetime format
data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S', errors='coerce')

# Extract the minutes since midnight from the 'arrival_time' column
data['minutes_since_midnight'] = data['arrival_time'].dt.hour * 60 + data['arrival_time'].dt.minute

# Aggregate the data by minutes since midnight and calculate the total number of passengers
data['total_passengers'] = data['passengers_up'] + data['passengers_continue']
minute_passengers = data.groupby('minutes_since_midnight')['total_passengers'].sum().reset_index()

# Convert minutes since midnight to hours for the x-axis labels
minute_passengers['hours'] = minute_passengers['minutes_since_midnight'] / 60

# Part 1: Identifying peak rush hours using minutes since midnight
# Plot the total number of passengers for each minute of the day
plt.figure(figsize=(12, 6))
plt.plot(minute_passengers['hours'], minute_passengers['total_passengers'], linestyle='-')
plt.title('Total Number of Passengers by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Number of Passengers')
plt.grid(True)
plt.xticks(range(0, 24))
plt.show()

# Part 2: Differences in bus usage between various regions
# Extract the hour from the 'arrival_time' column for the regional comparison
data['hour'] = data['arrival_time'].dt.hour

# Aggregate the data by region (cluster) and hour, then calculate the total number of passengers
region_hourly_passengers = data.groupby(['cluster', 'hour'])['total_passengers'].sum().reset_index()

# Create a pivot table for easier plotting
pivot_table = region_hourly_passengers.pivot(index='hour', columns='cluster', values='total_passengers')

# Plot the total number of passengers for each region by hour of the day
plt.figure(figsize=(14, 8))
pivot_table.plot(kind='line', marker='o', linestyle='-', figsize=(14, 8))

plt.title('Total Number of Passengers by Hour of the Day for Different Regions')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Number of Passengers')
plt.grid(True)
plt.xticks(range(0, 24))
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#Aggregate the total number of passengers for each region
region_passengers = data.groupby('cluster')['total_passengers'].sum().reset_index()

# Count the number of different lines operating in each region
region_lines = data.groupby('cluster')['line_id'].nunique().reset_index()

# Merge the two dataframes on the region (cluster) column
region_comparison = pd.merge(region_passengers, region_lines, on='cluster')
region_comparison.columns = ['Region', 'Total Passengers', 'Number of Lines']

# Plot the data
fig, ax1 = plt.subplots(figsize=(14, 8))

color = 'tab:blue'
ax1.set_xlabel('Region')
ax1.set_ylabel('Total Passengers', color=color)
ax1.bar(region_comparison['Region'], region_comparison['Total Passengers'], color=color, alpha=0.6, label='Total Passengers')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(region_comparison['Region'], rotation=45, ha='right')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Number of Lines', color=color)
ax2.plot(region_comparison['Region'], region_comparison['Number of Lines'], color=color, marker='o', label='Number of Lines')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the plot to fit the title
plt.title('Public Transportation Usage and Number of Different Lines in Each Region')
plt.show()

# Calculate the total number of passengers for each line
line_total_passengers = data.groupby('line_id')['total_passengers'].sum().reset_index()
top_n_lines = line_total_passengers.nlargest(10, 'total_passengers')['line_id']

# Filter the data to include only the top N lines
top_n_data = data[data['line_id'].isin(top_n_lines)]

# Aggregate data by line and hour to get the total passengers
top_n_utilization = top_n_data.groupby(['line_id', 'hour'])['total_passengers'].sum().reset_index()

# Plot utilization for the top N lines over time
plt.figure(figsize=(14, 8))
for line in top_n_utilization['line_id'].unique():
    line_data = top_n_utilization[top_n_utilization['line_id'] == line]
    plt.plot(line_data['hour'], line_data['total_passengers'], label=f'Line {line}')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Passengers')
plt.title('Top 10 Line Utilization Over Time')
plt.legend()
plt.grid(True)
plt.show()
