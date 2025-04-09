/**
 * Simulation JavaScript for UAV Path Planning
 * Handles the interactive simulation UI and communication with the server
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const simulationForm = document.getElementById('simulationForm');
    const startBtn = document.getElementById('startBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const resetBtn = document.getElementById('resetBtn');
    const simulationProgress = document.getElementById('simulationProgress');
    const currentTime = document.getElementById('currentTime');
    const energyRemaining = document.getElementById('energyRemaining');
    const flightDistance = document.getElementById('flightDistance');
    const servicedTasks = document.getElementById('servicedTasks');
    const activeUsers = document.getElementById('activeUsers');
    const currentService = document.getElementById('currentService');
    const trajectoryImage = document.getElementById('trajectoryImage');
    const energyImage = document.getElementById('energyImage');
    const metricDistance = document.getElementById('metricDistance');
    const metricEnergy = document.getElementById('metricEnergy');
    const metricTasks = document.getElementById('metricTasks');
    const metricDelay = document.getElementById('metricDelay');
    const metricEfficiency = document.getElementById('metricEfficiency');
    
    // Simulation state
    let simulationId = null;
    let simulationStatus = 'idle';
    let updateInterval = null;
    let imageUpdateInterval = null;
    
    // Function to start a new simulation
    async function startSimulation() {
        // Get form values
        const algorithm = document.getElementById('algorithm').value;
        const numUsers = document.getElementById('numUsers').value;
        const simTime = document.getElementById('simTime').value;
        const uavSpeed = document.getElementById('uavSpeed').value;
        
        // Disable start button, enable pause button
        startBtn.disabled = true;
        pauseBtn.disabled = false;
        resetBtn.disabled = true;
        
        try {
            // Call API to start simulation
            const response = await fetch('/api/start_simulation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    algorithm,
                    num_users: numUsers,
                    sim_time: simTime,
                    uav_speed: uavSpeed
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to start simulation');
            }
            
            const data = await response.json();
            simulationId = data.sim_id;
            simulationStatus = 'running';
            
            // Start polling for updates
            startUpdatePolling();
            
        } catch (error) {
            console.error('Error starting simulation:', error);
            alert('Error starting simulation: ' + error.message);
            
            // Reset buttons
            startBtn.disabled = false;
            pauseBtn.disabled = true;
            resetBtn.disabled = false;
        }
    }
    
    // Function to pause the simulation
    async function pauseSimulation() {
        if (!simulationId) return;
        
        try {
            const response = await fetch(`/api/pause_simulation/${simulationId}`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to pause simulation');
            }
            
            simulationStatus = 'paused';
            pauseBtn.textContent = 'Resume';
            pauseBtn.innerHTML = '<i data-feather="play"></i> Resume';
            feather.replace();
            
            // Stop regular polling and just update once
            clearInterval(updateInterval);
            clearInterval(imageUpdateInterval);
            updateSimulationState();
            
        } catch (error) {
            console.error('Error pausing simulation:', error);
            alert('Error pausing simulation: ' + error.message);
        }
    }
    
    // Function to resume the simulation
    async function resumeSimulation() {
        if (!simulationId) return;
        
        try {
            const response = await fetch(`/api/resume_simulation/${simulationId}`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to resume simulation');
            }
            
            simulationStatus = 'running';
            pauseBtn.textContent = 'Pause';
            pauseBtn.innerHTML = '<i data-feather="pause"></i> Pause';
            feather.replace();
            
            // Restart polling
            startUpdatePolling();
            
        } catch (error) {
            console.error('Error resuming simulation:', error);
            alert('Error resuming simulation: ' + error.message);
        }
    }
    
    // Function to reset the simulation
    function resetSimulation() {
        // Stop polling and clear state
        clearInterval(updateInterval);
        clearInterval(imageUpdateInterval);
        simulationId = null;
        simulationStatus = 'idle';
        
        // Reset UI elements
        simulationProgress.style.width = '0%';
        simulationProgress.textContent = '0%';
        currentTime.textContent = '0.0 s';
        energyRemaining.textContent = '10000.0 J';
        flightDistance.textContent = '0.0 m';
        servicedTasks.textContent = '0';
        activeUsers.textContent = '0';
        currentService.textContent = 'None';
        
        // Reset metrics
        metricDistance.textContent = '-';
        metricEnergy.textContent = '-';
        metricTasks.textContent = '-';
        metricDelay.textContent = '-';
        metricEfficiency.textContent = '-';
        
        // Reset images
        trajectoryImage.src = '/static/placeholder.svg';
        energyImage.src = '/static/placeholder.svg';
        
        // Enable/disable buttons
        startBtn.disabled = false;
        pauseBtn.disabled = true;
        resetBtn.disabled = false;
    }
    
    // Function to poll for simulation updates
    function startUpdatePolling() {
        // Clear existing intervals
        clearInterval(updateInterval);
        clearInterval(imageUpdateInterval);
        
        // Set update interval
        updateInterval = setInterval(updateSimulationState, 500);
        
        // Set image update interval (less frequent)
        imageUpdateInterval = setInterval(updateImages, 2000);
    }
    
    // Function to update simulation state
    async function updateSimulationState() {
        if (!simulationId) return;
        
        try {
            const response = await fetch(`/api/simulation_status/${simulationId}`);
            
            if (!response.ok) {
                throw new Error('Failed to get simulation status');
            }
            
            const data = await response.json();
            updateUIWithData(data);
            
            // Check if simulation is completed
            if (data.status === 'completed') {
                simulationStatus = 'completed';
                clearInterval(updateInterval);
                startBtn.disabled = false;
                pauseBtn.disabled = true;
                resetBtn.disabled = false;
                
                // Final image update
                updateImages();
            }
            
        } catch (error) {
            console.error('Error updating simulation state:', error);
            // Don't show alert to avoid spamming
            
            // Stop polling on error
            clearInterval(updateInterval);
            clearInterval(imageUpdateInterval);
            startBtn.disabled = false;
            pauseBtn.disabled = true;
            resetBtn.disabled = false;
        }
    }
    
    // Function to update UI with simulation data
    function updateUIWithData(data) {
        // Update progress
        const progress = data.progress.toFixed(1);
        simulationProgress.style.width = `${progress}%`;
        simulationProgress.textContent = `${progress}%`;
        simulationProgress.setAttribute('aria-valuenow', progress);
        
        // Update status information
        if (data.current_state.time !== undefined) {
            currentTime.textContent = `${data.current_state.time.toFixed(1)} s`;
        }
        
        if (data.current_state.uav_energy !== undefined) {
            energyRemaining.textContent = `${data.current_state.uav_energy.toFixed(1)} J`;
            
            // Change color based on energy level
            const energyPercent = data.current_state.uav_energy / 10000.0 * 100; // Assuming 10000 is initial energy
            if (energyPercent < 20) {
                energyRemaining.className = 'badge bg-danger rounded-pill';
            } else if (energyPercent < 50) {
                energyRemaining.className = 'badge bg-warning rounded-pill';
            } else {
                energyRemaining.className = 'badge bg-success rounded-pill';
            }
        }
        
        if (data.current_state.total_flight_distance !== undefined) {
            flightDistance.textContent = `${data.current_state.total_flight_distance.toFixed(1)} m`;
        }
        
        if (data.current_state.serviced_tasks !== undefined) {
            servicedTasks.textContent = data.current_state.serviced_tasks;
        }
        
        if (data.current_state.users_with_tasks !== undefined) {
            activeUsers.textContent = data.current_state.users_with_tasks.length;
        }
        
        if (data.current_state.current_user !== undefined) {
            currentService.textContent = data.current_state.current_user !== null 
                ? `User ${data.current_state.current_user}` 
                : 'None';
        }
        
        // Update metrics if simulation is completed
        if (data.status === 'completed' && data.metrics) {
            metricDistance.textContent = `${data.metrics.total_flight_distance.toFixed(2)} m`;
            metricEnergy.textContent = `${data.metrics.energy_consumed.toFixed(2)} J`;
            metricTasks.textContent = data.metrics.serviced_tasks;
            metricDelay.textContent = `${data.metrics.avg_task_delay.toFixed(2)} s`;
            metricEfficiency.textContent = `${data.metrics.energy_efficiency.toFixed(4)} bits/J`;
        }
    }
    
    // Function to update images
    function updateImages() {
        if (!simulationId) return;
        
        // Add timestamp to prevent caching
        const timestamp = Date.now();
        trajectoryImage.src = `/api/get_trajectory/${simulationId}?t=${timestamp}`;
        energyImage.src = `/api/get_energy/${simulationId}?t=${timestamp}`;
    }
    
    // Event listeners
    startBtn.addEventListener('click', function(e) {
        e.preventDefault();
        startSimulation();
    });
    
    pauseBtn.addEventListener('click', function(e) {
        e.preventDefault();
        if (simulationStatus === 'running') {
            pauseSimulation();
        } else if (simulationStatus === 'paused') {
            resumeSimulation();
        }
    });
    
    resetBtn.addEventListener('click', function(e) {
        e.preventDefault();
        resetSimulation();
    });
    
    simulationForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // If simulation is running, apply settings but don't restart
        if (simulationStatus === 'running' || simulationStatus === 'paused') {
            // Currently we can't change settings mid-simulation
            alert('Please reset the simulation before changing settings.');
            return;
        }
        
        // Otherwise, start a new simulation
        startSimulation();
    });
});