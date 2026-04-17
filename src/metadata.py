import datetime

def parse_metadata(filepath):
    """
    Parses metadata.txt with format "dd/mm/yyyy hh:mm, unsaved"
    Returns a score between 0 and 1, and an inferences string
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            
        if not content:
            return 0.0, "Empty metadata"
            
        parts = content.split(",")
        if len(parts) != 2:
            return 0.0, f"Unrecognized metadata format: {content}"
            
        datetime_str = parts[0].strip()
        status_str = parts[1].strip().lower()
        
        # Parse datetime
        # Expected format: dd/mm/yyyy hh:mm
        dt = datetime.datetime.strptime(datetime_str, "%d/%m/%Y %H:%M")
        
        score = 0.0
        inferences = []
        
        # Check if unsaved
        if status_str == "unsaved":
            score += 0.5
            inferences.append("Unsaved contact")
        else:
            inferences.append("Saved contact")
            
        # Check time of day (late night is suspicious: e.g. 23:00 to 05:00)
        hour = dt.hour
        if hour >= 23 or hour <= 5:
            score += 0.5
            inferences.append(f"Late night call ({dt.strftime('%H:%M')})")
        else:
            inferences.append(f"Normal hours ({dt.strftime('%H:%M')})")
            
        return score, ", ".join(inferences)
        
    except Exception as e:
        return 0.0, f"Metadata parse error: {str(e)}"

if __name__ == "__main__":
    # Test script
    import os
    test_file = "dummy_metadata.txt"
    with open(test_file, 'w') as f:
        f.write("12/03/2026 23:45, unsaved")
    
    score, inf = parse_metadata(test_file)
    print(f"Metadata Score: {score}")
    print(f"Metadata Inferences: {inf}")
    os.remove(test_file)
