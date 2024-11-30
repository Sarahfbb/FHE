import numpy as np
import os

def combine_results():
    # Load predictions and execution times
    pred_0_50 = np.load('predictions_range_0_50.npy')
    pred_50_100 = np.load('predictions_range_50_100.npy')
    exec_times_0_50 = np.load('execution_times_range_0_50.npy')
    exec_times_50_100 = np.load('execution_times_range_50_100.npy')

    # Combine predictions and execution times
    all_predictions = np.vstack((pred_0_50, pred_50_100))
    all_execution_times = np.concatenate((exec_times_0_50, exec_times_50_100))

    # Load test labels
    test_labels = np.load('features/1TestT.npy')

    # Calculate overall accuracy
    predicted_labels = np.argmax(all_predictions, axis=1)
    overall_accuracy = np.mean(predicted_labels == test_labels)

    # Calculate average execution time
    avg_execution_time = np.mean(all_execution_times)

    # Print results
    print(f"Overall test accuracy: {overall_accuracy:.4f}")
    print(f"Average execution time: {avg_execution_time:.2f} seconds per sample")
    print(f"Total samples processed: {len(predicted_labels)}")

    # Optional: Save combined results
    np.save('combined_predictions.npy', all_predictions)
    np.save('combined_execution_times.npy', all_execution_times)

    # Additional statistics
    print(f"\nExecution time statistics:")
    print(f"Minimum: {np.min(all_execution_times):.2f} seconds")
    print(f"Maximum: {np.max(all_execution_times):.2f} seconds")
    print(f"Median: {np.median(all_execution_times):.2f} seconds")
    print(f"Standard deviation: {np.std(all_execution_times):.2f} seconds")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels, predicted_labels)
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    combine_results()