import axios from "axios";

const API_BASE = "http://127.0.0.1:5000";

export const fetchBestLocations = async (city, business) => {
  const response = await axios.post(`${API_BASE}/api/location`, {
    city: city,
    business: business,
  });

  return response.data; 
};
