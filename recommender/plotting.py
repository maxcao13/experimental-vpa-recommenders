import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from recommender.recommender_config import PLOT_TIME_RANGE

def plot_resource_usage(cpu_data, memory_data, container_name, model_trainer, time_range=PLOT_TIME_RANGE):
    """
    Plot CPU and memory usage history with enhanced visualization and prediction tracking
    time_range: string specifying the time range to plot ('5m', '1h', '2h', '6h', '12h', '1d', '7d')
    """
    # Convert time range string to timedelta
    time_ranges = {
        '5m': timedelta(minutes=5),
        '1h': timedelta(hours=1),
        '2h': timedelta(hours=2),
        '6h': timedelta(hours=6),
        '12h': timedelta(hours=12),
        '1d': timedelta(days=1),
        '7d': timedelta(days=7)
    }
    
    if time_range not in time_ranges:
        print(f"Warning: Invalid time range '{time_range}'. Using default '7d'")
        time_range = '7d'
    
    delta = time_ranges[time_range]
    cutoff_time = datetime.now() - delta
    current_time = datetime.now()
    
    # Convert Prometheus data to DataFrame and sort by timestamp
    cpu_df = pd.DataFrame([
        {'timestamp': datetime.fromtimestamp(float(point[0])), 
         'value': float(point[1]) * 1000,  # Convert to millicores
         'pod': metric.get('metric', {}).get('pod', 'unknown')}  # Add pod name
        for metric in cpu_data 
        for point in metric['values']
    ]).sort_values('timestamp')
    
    memory_df = pd.DataFrame([
        {'timestamp': datetime.fromtimestamp(float(point[0])), 
         'value': float(point[1]) / (1024 * 1024),  # Convert to MB
         'pod': metric.get('metric', {}).get('pod', 'unknown')}  # Add pod name
        for metric in memory_data 
        for point in metric['values']
    ]).sort_values('timestamp')

    # Filter data to time range
    cpu_df = cpu_df[cpu_df['timestamp'] >= cutoff_time]
    memory_df = memory_df[memory_df['timestamp'] >= cutoff_time]

    # Get past predictions
    cpu_predictions = model_trainer.get_predictions('cpu')
    memory_predictions = model_trainer.get_predictions('memory')

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot CPU usage with unique color per pod
    for pod in cpu_df['pod'].unique():
        pod_data = cpu_df[cpu_df['pod'] == pod]
        ax1.plot(pod_data['timestamp'], pod_data['value'], '-', linewidth=0.8, alpha=0.6, label=f'Pod: {pod}')  # Thinner, slightly transparent line
        ax1.scatter(pod_data['timestamp'], pod_data['value'], s=20, alpha=0.8)  # Add points
    
    # Plot CPU predictions if available
    if cpu_predictions['timestamps'] and any(v is not None for v in cpu_predictions['values']):
        pred_df = pd.DataFrame({
            'timestamp': cpu_predictions['timestamps'],
            'value': [v * 1000 if v is not None else None for v in cpu_predictions['values']]  # Convert to millicores
        }).dropna()  # Remove any None values
        
        if not pred_df.empty:
            pred_df = pred_df[pred_df['timestamp'] >= cutoff_time]
            if not pred_df.empty:
                # Split predictions into past and future
                past_preds = pred_df[pred_df['timestamp'] <= current_time]
                future_preds = pred_df[pred_df['timestamp'] > current_time]
                
                # Plot past predictions
                if not past_preds.empty:
                    ax1.plot(past_preds['timestamp'], past_preds['value'], '--', color='#00FF00', linewidth=1, alpha=0.7, label='Past Predictions')
                    ax1.scatter(past_preds['timestamp'], past_preds['value'], color='#00FF00', s=30, alpha=0.8)
                
                # Plot future predictions
                if not future_preds.empty:
                    ax1.plot(future_preds['timestamp'], future_preds['value'], '--', color='#FF00FF', linewidth=1, alpha=0.7, label='Future Predictions')
                    ax1.scatter(future_preds['timestamp'], future_preds['value'], color='#FF00FF', s=30, alpha=0.8)
                
                # Calculate prediction accuracy metrics for past predictions
                if not past_preds.empty:
                    visible_mask = (past_preds['timestamp'] >= cpu_df['timestamp'].min()) & (past_preds['timestamp'] <= cpu_df['timestamp'].max())
                    if any(visible_mask):
                        visible_preds = past_preds[visible_mask]
                        # Interpolate actual values to prediction timestamps
                        actual_values = np.interp(
                            [t.timestamp() for t in visible_preds['timestamp']], 
                            [t.timestamp() for t in cpu_df['timestamp']], 
                            cpu_df['value']
                        )
                        mape = np.mean(np.abs((actual_values - visible_preds['value']) / actual_values)) * 100
                        rmse = np.sqrt(np.mean((actual_values - visible_preds['value']) ** 2))
                        
                        accuracy_text = f"Past Prediction Metrics:\nMAPE: {mape:.1f}%\nRMSE: {rmse:.1f}m"
                        ax1.text(0.02, 0.85, accuracy_text, transform=ax1.transAxes, 
                                verticalalignment='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add vertical line for current time
    ax1.axvline(x=current_time, color='gray', linestyle=':', alpha=0.5, label='Current Time')
    
    ax1.set_title(f'CPU Usage Over Time - {container_name} (Last {time_range})')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('CPU (millicores)')
    ax1.grid(True)
    ax1.legend()
    
    # Add CPU statistics
    cpu_stats = (f"CPU Stats - Mean: {cpu_df['value'].mean():.1f}m, "
                f"Max: {cpu_df['value'].max():.1f}m, "
                f"Min: {cpu_df['value'].min():.1f}m")
    ax1.text(0.02, 0.98, cpu_stats, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=8)
    
    # Plot Memory usage with unique color per pod
    for pod in memory_df['pod'].unique():
        pod_data = memory_df[memory_df['pod'] == pod]
        ax2.plot(pod_data['timestamp'], pod_data['value'], '-', linewidth=0.8, alpha=0.6, label=f'Pod: {pod}')  # Thinner, slightly transparent line
        ax2.scatter(pod_data['timestamp'], pod_data['value'], s=20, alpha=0.8)  # Add points
    
    # Plot memory predictions if available
    if memory_predictions['timestamps'] and any(v is not None for v in memory_predictions['values']):
        pred_df = pd.DataFrame({
            'timestamp': memory_predictions['timestamps'],
            'value': [v / (1024 * 1024) if v is not None else None for v in memory_predictions['values']]  # Convert to MB
        }).dropna()  # Remove any None values
        
        if not pred_df.empty:
            pred_df = pred_df[pred_df['timestamp'] >= cutoff_time]
            if not pred_df.empty:
                # Split predictions into past and future
                past_preds = pred_df[pred_df['timestamp'] <= current_time]
                future_preds = pred_df[pred_df['timestamp'] > current_time]
                
                # Plot past predictions
                if not past_preds.empty:
                    ax2.plot(past_preds['timestamp'], past_preds['value'], '--', color='#00FF00', linewidth=1, alpha=0.7, label='Past Predictions')
                    ax2.scatter(past_preds['timestamp'], past_preds['value'], color='#00FF00', s=30, alpha=0.8)
                
                # Plot future predictions
                if not future_preds.empty:
                    ax2.plot(future_preds['timestamp'], future_preds['value'], '--', color='#FF00FF', linewidth=1, alpha=0.7, label='Future Predictions')
                    ax2.scatter(future_preds['timestamp'], future_preds['value'], color='#FF00FF', s=30, alpha=0.8)
                
                # Calculate prediction accuracy metrics for past predictions
                if not past_preds.empty:
                    visible_mask = (past_preds['timestamp'] >= memory_df['timestamp'].min()) & (past_preds['timestamp'] <= memory_df['timestamp'].max())
                    if any(visible_mask):
                        visible_preds = past_preds[visible_mask]
                        # Interpolate actual values to prediction timestamps
                        actual_values = np.interp(
                            [t.timestamp() for t in visible_preds['timestamp']], 
                            [t.timestamp() for t in memory_df['timestamp']], 
                            memory_df['value']
                        )
                        mape = np.mean(np.abs((actual_values - visible_preds['value']) / actual_values)) * 100
                        rmse = np.sqrt(np.mean((actual_values - visible_preds['value']) ** 2))
                        
                        accuracy_text = f"Past Prediction Metrics:\nMAPE: {mape:.1f}%\nRMSE: {rmse:.1f}MB"
                        ax2.text(0.02, 0.85, accuracy_text, transform=ax2.transAxes, 
                                verticalalignment='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add vertical line for current time
    ax2.axvline(x=current_time, color='gray', linestyle=':', alpha=0.5, label='Current Time')
    
    ax2.set_title(f'Memory Usage Over Time - {container_name} (Last {time_range})')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Memory (MB)')
    ax2.grid(True)
    ax2.legend()
    
    # Add Memory statistics
    mem_stats = (f"Memory Stats - Mean: {memory_df['value'].mean():.1f}MB, "
                f"Max: {memory_df['value'].max():.1f}MB, "
                f"Min: {memory_df['value'].min():.1f}MB")
    ax2.text(0.02, 0.98, mem_stats, transform=ax2.transAxes, 
             verticalalignment='top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'resource_usage_{container_name}_{time_range}.png')
    plt.close() 