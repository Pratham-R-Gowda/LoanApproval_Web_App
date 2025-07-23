const { spawn } = require('child_process');
const express = require('express');
const cors = require('cors');
const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

app.post('/predict', (req, res) => {
  const inputData = JSON.stringify(req.body);

  const python = spawn('python', ['predict.py']); // or 'python3' if required

  let pythonData = '';

  // Write input data to the Python script
  python.stdin.write(inputData);
  python.stdin.end(); // Important: close stdin so Python knows it's done

  // Read output from Python
  python.stdout.on('data', (data) => {
    pythonData += data.toString();
  });

  // On close, parse Python output and send to client
  python.on('close', (code) => {
    try {
      const result = JSON.parse(pythonData);
      res.json(result);
    } catch (err) {
      console.error('Error parsing Python output:', err);
      console.error('Raw Python Output:', pythonData);
      res.status(500).json({ error: 'Failed to parse prediction result' });
    }
  });

  // Handle Python errors
  python.stderr.on('data', (data) => {
    console.error(`Python error: ${data.toString()}`);
  });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
