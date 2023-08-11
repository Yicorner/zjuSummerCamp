import timeit

class TestTime:
    def __init__(self):
        self.segments = {}
        self.start_times = {}  # To track start times of all segments

    def start(self, segment_name):
        """Start the timer for a specific segment."""
        if segment_name in self.start_times:
            raise Exception(f"Segment {segment_name} is already running. Stop it before starting again.")

        self.start_times[segment_name] = timeit.default_timer()

    def stop(self, segment_name):
        """Stop the timer for a specific segment."""
        if segment_name not in self.start_times:
            raise Exception(f"No timer started for segment: {segment_name}")

        end_time = timeit.default_timer()
        elapsed_time = end_time - self.start_times[segment_name]

        self.segments[segment_name] = elapsed_time

        # Removing start time for the segment after stopping
        del self.start_times[segment_name]

    def get_time(self, segment_name):
        """Retrieve the time for a specific segment."""
        return self.segments.get(segment_name)

    def report(self):
        """Print a report of all segment times."""
        for segment, runtime in self.segments.items():
            print(f"Segment: {segment}, Time elapsed: {runtime:.6f} seconds")  # Increased precision

# Example usage
if __name__ == "__main__":
    timer = TestTime()

    timer.start("Segment1")
    timer.start("Segment2")
    timeit.time.sleep(2)  # Simulating some task using timeit's time for consistency
    timer.stop("Segment2")

    timeit.time.sleep(3)  # Simulating some task using timeit's time for consistency
    timer.stop("Segment1")

    timer.report()