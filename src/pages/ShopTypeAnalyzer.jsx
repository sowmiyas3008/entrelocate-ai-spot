// import { useState } from "react";
// import { Building2, TrendingUp, MapPin, Loader2, AlertCircle } from "lucide-react";

// // FindShopType Component
// const FindShopType = ({ onAnalyze }) => {
//   const [city, setCity] = useState("");

//   const handleSubmit = (e) => {
//     e.preventDefault();
//     if (city.trim()) {
//       onAnalyze(city);
//     }
//   };

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-4 py-12">
//       <div className="container mx-auto max-w-2xl">
//         <div className="bg-white rounded-2xl shadow-xl p-8 animate-fade-in">
//           <div className="flex items-center gap-3 mb-6">
//             <Building2 className="h-10 w-10 text-blue-600" />
//             <div>
//               <h1 className="text-3xl font-bold text-gray-900">Find Best Shop Type</h1>
//               <p className="text-gray-600">For your chosen city</p>
//             </div>
//           </div>

//           <div className="space-y-6">
//             <div>
//               <label htmlFor="city" className="block text-sm font-medium text-gray-700 mb-2">
//                 City
//               </label>
//               <input
//                 id="city"
//                 type="text"
//                 placeholder="e.g., Puducherry, Chennai, Bangalore"
//                 value={city}
//                 onChange={(e) => setCity(e.target.value)}
//                 onKeyPress={(e) => e.key === 'Enter' && handleSubmit(e)}
//                 className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
//               />
//               <p className="text-sm text-gray-500 mt-2">
//                 We'll analyze market trends and demographics to suggest the best business types
//               </p>
//             </div>

//             <button
//               onClick={handleSubmit}
//               className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors font-medium"
//             >
//               Analyze City
//             </button>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// // ShopTypeResults Component
// const ShopTypeResults = ({ city, results, onNewSearch }) => {
//   if (!results) return null;

//   const getPotentialColor = (score) => {
//     if (score >= 70) return "text-green-600";
//     if (score >= 50) return "text-yellow-600";
//     return "text-orange-600";
//   };

//   const getPotentialLabel = (score) => {
//     if (score >= 70) return "Very High";
//     if (score >= 50) return "High";
//     if (score >= 30) return "Medium";
//     return "Low";
//   };

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-4 py-12">
//       <div className="container mx-auto max-w-4xl">
//         <div className="mb-8 animate-fade-in">
//           <h1 className="text-4xl font-bold mb-2 text-gray-900">
//             Top Business Types in {city}
//           </h1>
//           <p className="text-xl text-gray-600">Based on market gap analysis</p>
//         </div>

//         <div className="space-y-4 mb-8">
//           {results.map((shop, index) => (
//             <div
//               key={index}
//               className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-all"
//               style={{ animation: `slideUp 0.5s ease-out ${index * 0.1}s both` }}
//             >
//               <div className="flex items-start justify-between">
//                 <div className="flex-1">
//                   <div className="flex items-center gap-2 mb-3">
//                     <Building2 className="h-6 w-6 text-blue-600" />
//                     <h3 className="text-2xl font-bold text-gray-900">{shop.category}</h3>
//                   </div>

//                   <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm mb-4">
//                     <div className="flex items-center gap-2">
//                       <TrendingUp className="h-5 w-5 text-green-600" />
//                       <span className="text-gray-600">
//                         Market Potential:{" "}
//                         <strong className={getPotentialColor(shop.max_score)}>
//                           {getPotentialLabel(shop.max_score)} ({shop.max_score}/100)
//                         </strong>
//                       </span>
//                     </div>
                    
//                     <div className="flex items-center gap-2">
//                       <MapPin className="h-5 w-5 text-blue-600" />
//                       <span className="text-gray-600">
//                         Best Cluster: <strong className="text-gray-900">#{shop.max_score_cluster}</strong>
//                       </span>
//                     </div>
//                   </div>

//                   {shop.neighborhood && shop.neighborhood.length > 0 && (
//                     <div className="bg-blue-50 rounded-lg p-4">
//                       <h4 className="text-sm font-semibold text-gray-900 mb-2">
//                         Recommended Neighborhoods:
//                       </h4>
//                       <div className="flex flex-wrap gap-2">
//                         {shop.neighborhood.map((area, i) => (
//                           <span
//                             key={i}
//                             className="bg-white text-gray-700 px-3 py-1 rounded-full text-sm border border-gray-200"
//                           >
//                             {area}
//                           </span>
//                         ))}
//                       </div>
//                     </div>
//                   )}
//                 </div>

//                 <div className="ml-4">
//                   <div className="text-right">
//                     <div className="text-3xl font-bold text-blue-600">
//                       #{index + 1}
//                     </div>
//                     <div className="text-xs text-gray-500">Rank</div>
//                   </div>
//                 </div>
//               </div>
//             </div>
//           ))}
//         </div>

//         <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-blue-100">
//           <h3 className="text-xl font-bold mb-3 text-gray-900">💡 Understanding the Results</h3>
//           <p className="text-gray-600 mb-4">
//             These recommendations are based on market gap analysis - areas where there's demand but low supply of these business types.
//           </p>
//           <ul className="space-y-2 text-gray-600">
//             <li className="flex items-start gap-2">
//               <span className="text-blue-600 font-bold">•</span>
//               <span><strong>High Potential:</strong> Low competition, strong demand indicators</span>
//             </li>
//             <li className="flex items-start gap-2">
//               <span className="text-blue-600 font-bold">•</span>
//               <span><strong>Clusters:</strong> Geographic areas identified through analysis</span>
//             </li>
//             <li className="flex items-start gap-2">
//               <span className="text-blue-600 font-bold">•</span>
//               <span><strong>Neighborhoods:</strong> Specific areas within the city to explore</span>
//             </li>
//           </ul>
//         </div>

//         <div className="mt-8 flex gap-4">
//           <button
//             onClick={onNewSearch}
//             className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors font-medium"
//           >
//             Search Another City
//           </button>
//         </div>
//       </div>
//     </div>
//   );
// };

// // Loading Component
// const LoadingScreen = ({ city }) => (
//   <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center px-4">
//     <div className="bg-white rounded-2xl shadow-xl p-12 text-center max-w-md w-full">
//       <Loader2 className="h-16 w-16 text-blue-600 animate-spin mx-auto mb-6" />
//       <h2 className="text-2xl font-bold text-gray-900 mb-2">Analyzing {city}...</h2>
//       <p className="text-gray-600 mb-6">
//         Gathering shop data, analyzing market gaps, and identifying opportunities
//       </p>
//       <div className="space-y-2 text-left text-sm text-gray-600">
//         <div className="flex items-center gap-2">
//           <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
//           Fetching business data from OpenStreetMap
//         </div>
//         <div className="flex items-center gap-2">
//           <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
//           Performing clustering analysis
//         </div>
//         <div className="flex items-center gap-2">
//           <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
//           Identifying market opportunities
//         </div>
//       </div>
//     </div>
//   </div>
// );

// // Error Component
// const ErrorScreen = ({ error, onRetry }) => (
//   <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center px-4">
//     <div className="bg-white rounded-2xl shadow-xl p-12 text-center max-w-md w-full">
//       <AlertCircle className="h-16 w-16 text-red-600 mx-auto mb-6" />
//       <h2 className="text-2xl font-bold text-gray-900 mb-2">Analysis Failed</h2>
//       <p className="text-gray-600 mb-6">{error}</p>
//       <button
//         onClick={onRetry}
//         className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors font-medium"
//       >
//         Try Again
//       </button>
//     </div>
//   </div>
// );

// // Main App Component
// const ShopTypeAnalyzer = () => {
//   const [view, setView] = useState('input'); // 'input', 'loading', 'results', 'error'
//   const [city, setCity] = useState('');
//   const [results, setResults] = useState(null);
//   const [error, setError] = useState('');

//   const analyzeCity = async (cityName) => {
//     setCity(cityName);
//     setView('loading');
//     setError('');

//     try {
//       const response = await fetch('http://localhost:5000/api/business', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ city: cityName }),
//       });

//       if (!response.ok) {
//         const errorData = await response.json();
//         throw new Error(errorData.error || 'Failed to analyze city');
//       }

//       const data = await response.json();
      
//       if (!data || data.length === 0) {
//         throw new Error('No business opportunities found for this city');
//       }

//       setResults(data);
//       setView('results');
//     } catch (err) {
//       console.error('Error:', err);
//       setError(err.message || 'Failed to analyze city. Please try again.');
//       setView('error');
//     }
//   };

//   const handleNewSearch = () => {
//     setView('input');
//     setCity('');
//     setResults(null);
//     setError('');
//   };

//   const handleRetry = () => {
//     if (city) {
//       analyzeCity(city);
//     } else {
//       setView('input');
//     }
//   };

//   return (
//     <>
//       <style>{`
//         @keyframes slideUp {
//           from {
//             opacity: 0;
//             transform: translateY(20px);
//           }
//           to {
//             opacity: 1;
//             transform: translateY(0);
//           }
//         }
//         @keyframes fade-in {
//           from { opacity: 0; }
//           to { opacity: 1; }
//         }
//         .animate-fade-in {
//           animation: fade-in 0.5s ease-out;
//         }
//       `}</style>

//       {view === 'input' && <FindShopType onAnalyze={analyzeCity} />}
//       {view === 'loading' && <LoadingScreen city={city} />}
//       {view === 'results' && (
//         <ShopTypeResults
//           city={city}
//           results={results}
//           onNewSearch={handleNewSearch}
//         />
//       )}
//       {view === 'error' && <ErrorScreen error={error} onRetry={handleRetry} />}
//     </>
//   );
// };

// export default ShopTypeAnalyzer;

import { useState, useEffect, useRef } from "react";
import { Building2, TrendingUp, MapPin, Loader2, AlertCircle, Map as MapIcon } from "lucide-react";

// FindShopType Component
const FindShopType = ({ onAnalyze, onBack }) => {
  const [city, setCity] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (city.trim()) {
      onAnalyze(city);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-4 py-12">
      <div className="container mx-auto max-w-3xl">
        <div className="bg-white rounded-2xl shadow-xl p-10 animate-fade-in">
          <div className="flex items-center gap-3 mb-6">
            <Building2 className="h-10 w-10 text-orange-600" />
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Find Best Shop Type</h1>
              <p className="text-gray-600">For your chosen city</p>
            </div>
          </div>

          <div className="space-y-6">
            <div>
              <label htmlFor="city" className="block text-sm font-medium text-gray-700 mb-2">
                City
              </label>
              <input
                id="city"
                type="text"
                placeholder="e.g., Puducherry, Chennai, Bangalore"
                value={city}
                onChange={(e) => setCity(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSubmit(e)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent"
              />
              <p className="text-sm text-gray-500 mt-2">
                We'll analyze market trends and demographics to suggest the best business types
              </p>
            </div>

            <div className="flex gap-4">
              <button
                onClick={onBack}
                className="flex-1 bg-gray-100 text-gray-700 py-3 px-6 rounded-lg hover:bg-gray-200 transition-colors font-medium border border-gray-300"
              >
                Back to Selection
              </button>
              <button
                onClick={handleSubmit}
                className="flex-1 bg-orange-600 text-white py-3 px-6 rounded-lg hover:bg-orange-700 transition-colors font-medium"
              >
                Analyze City
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// ClusterMap Component
const ClusterMap = ({ clusterData, city }) => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);

  useEffect(() => {
    // Load Leaflet CSS
    if (!document.getElementById('leaflet-css')) {
      const link = document.createElement('link');
      link.id = 'leaflet-css';
      link.rel = 'stylesheet';
      link.href = 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css';
      document.head.appendChild(link);
    }

    // Load Leaflet JS
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js';
    script.async = true;
    
    script.onload = () => {
      if (mapRef.current && window.L && !mapInstanceRef.current) {
        // Calculate center from all clusters
        const avgLat = clusterData.reduce((sum, c) => sum + c.lat, 0) / clusterData.length;
        const avgLon = clusterData.reduce((sum, c) => sum + c.lon, 0) / clusterData.length;

        // Initialize map
        const map = window.L.map(mapRef.current).setView([avgLat, avgLon], 12);
        mapInstanceRef.current = map;

        // Add tile layer
        window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '© OpenStreetMap contributors',
          maxZoom: 18
        }).addTo(map);

        // Color palette for different ranks
        const colors = ['#2563eb', '#7c3aed', '#db2777', '#ea580c', '#ca8a04'];

        // Add circles for each cluster
        clusterData.forEach((cluster, index) => {
          const color = colors[index % colors.length];
          
          // Add a circle marker
          const circle = window.L.circle([cluster.lat, cluster.lon], {
            color: color,
            fillColor: color,
            fillOpacity: 0.3,
            radius: 500, // 500 meters radius
            weight: 2
          }).addTo(map);

          // Add popup
          const popupContent = `
            <div style="font-family: system-ui, -apple-system, sans-serif;">
              <h3 style="font-weight: bold; margin: 0 0 8px 0; color: #1f2937;">
                #${index + 1} - ${cluster.category}
              </h3>
              <p style="margin: 4px 0; color: #4b5563;">
                <strong>Score:</strong> ${cluster.score}/100
              </p>
              <p style="margin: 4px 0; color: #4b5563;">
                <strong>Cluster:</strong> #${cluster.cluster_id}
              </p>
              ${cluster.neighborhoods && cluster.neighborhoods.length > 0 ? `
                <p style="margin: 4px 0; color: #4b5563;">
                  <strong>Areas:</strong><br/>
                  ${cluster.neighborhoods.slice(0, 3).join(', ')}
                </p>
              ` : ''}
            </div>
          `;
          
          circle.bindPopup(popupContent);

          // Add numbered marker
          const icon = window.L.divIcon({
            className: 'custom-div-icon',
            html: `<div style="
              background-color: ${color};
              color: white;
              border-radius: 50%;
              width: 32px;
              height: 32px;
              display: flex;
              align-items: center;
              justify-content: center;
              font-weight: bold;
              font-size: 14px;
              border: 3px solid white;
              box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            ">${index + 1}</div>`,
            iconSize: [32, 32],
            iconAnchor: [16, 16]
          });

          window.L.marker([cluster.lat, cluster.lon], { icon }).addTo(map);
        });

        // Fit bounds to show all markers
        const bounds = window.L.latLngBounds(
          clusterData.map(c => [c.lat, c.lon])
        );
        map.fitBounds(bounds, { padding: [50, 50] });
      }
    };

    document.body.appendChild(script);

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [clusterData]);

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden mb-8">
      <div className="p-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white">
        <div className="flex items-center gap-2">
          <MapIcon className="h-5 w-5" />
          <h3 className="text-lg font-bold">Cluster Locations in {city}</h3>
        </div>
        <p className="text-sm text-blue-100 mt-1">
          Click on markers to see details. Shaded areas show cluster coverage.
        </p>
      </div>
      <div ref={mapRef} style={{ height: '500px', width: '100%' }} />
    </div>
  );
};

// ShopTypeResults Component
const ShopTypeResults = ({ city, results, shopData, onNewSearch }) => {
  if (!results) return null;

  const getPotentialColor = (score) => {
    if (score >= 70) return "text-green-600";
    if (score >= 50) return "text-yellow-600";
    return "text-orange-600";
  };

  const getPotentialLabel = (score) => {
    if (score >= 70) return "Very High";
    if (score >= 50) return "High";
    if (score >= 30) return "Medium";
    return "Low";
  };

  // Prepare data for map
  const clusterMapData = results.map((shop, index) => ({
    category: shop.category,
    score: shop.max_score,
    cluster_id: shop.max_score_cluster,
    lat: shop.latitude || 0,
    lon: shop.longitude || 0,
    neighborhoods: shop.neighborhood || []
  })).filter(c => c.lat !== 0 && c.lon !== 0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-4 py-12">
      <div className="container mx-auto max-w-6xl">
        <div className="mb-8 animate-fade-in">
          <h1 className="text-4xl font-bold mb-2 text-gray-900">
            Top Business Types in {city}
          </h1>
          <p className="text-xl text-gray-600">Based on market gap analysis</p>
        </div>

        {clusterMapData.length > 0 && (
          <ClusterMap clusterData={clusterMapData} city={city} />
        )}

        <div className="space-y-4 mb-8">
          {results.map((shop, index) => (
            <div
              key={index}
              className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-all"
              style={{ animation: `slideUp 0.5s ease-out ${index * 0.1}s both` }}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-3">
                    <Building2 className="h-6 w-6 text-blue-600" />
                    <h3 className="text-2xl font-bold text-gray-900">{shop.category}</h3>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm mb-4">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5 text-green-600" />
                      <span className="text-gray-600">
                        Market Potential:{" "}
                        <strong className={getPotentialColor(shop.max_score)}>
                          {getPotentialLabel(shop.max_score)} ({shop.max_score}/100)
                        </strong>
                      </span>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <MapPin className="h-5 w-5 text-blue-600" />
                      <span className="text-gray-600">
                        Best Cluster: <strong className="text-gray-900">#{shop.max_score_cluster}</strong>
                      </span>
                    </div>
                  </div>

                  {shop.neighborhood && shop.neighborhood.length > 0 && (
                    <div className="bg-blue-50 rounded-lg p-4">
                      <h4 className="text-sm font-semibold text-gray-900 mb-2">
                        Recommended Neighborhoods:
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {shop.neighborhood.map((area, i) => (
                          <span
                            key={i}
                            className="bg-white text-gray-700 px-3 py-1 rounded-full text-sm border border-gray-200"
                          >
                            {area}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="ml-4">
                  <div className="text-right">
                    <div className="text-3xl font-bold text-blue-600">
                      #{index + 1}
                    </div>
                    <div className="text-xs text-gray-500">Rank</div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-blue-100">
          <h3 className="text-xl font-bold mb-3 text-gray-900">💡 Understanding the Results</h3>
          <p className="text-gray-600 mb-4">
            These recommendations are based on market gap analysis - areas where there's demand but low supply of these business types.
          </p>
          <ul className="space-y-2 text-gray-600">
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">•</span>
              <span><strong>High Potential:</strong> Low competition, strong demand indicators</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">•</span>
              <span><strong>Clusters:</strong> Geographic areas identified through analysis</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">•</span>
              <span><strong>Map Markers:</strong> Click on numbered circles to see details</span>
            </li>
          </ul>
        </div>

        <div className="mt-8 flex gap-4">
          <button
            onClick={onNewSearch}
            className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            Search Another City
          </button>
        </div>
      </div>
    </div>
  );
};

// Loading Component
const LoadingScreen = ({ city }) => (
  <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center px-4">
    <div className="bg-white rounded-2xl shadow-xl p-12 text-center max-w-md w-full">
      <Loader2 className="h-16 w-16 text-blue-600 animate-spin mx-auto mb-6" />
      <h2 className="text-2xl font-bold text-gray-900 mb-2">Analyzing {city}...</h2>
      <p className="text-gray-600 mb-6">
        Gathering shop data, analyzing market gaps, and identifying opportunities
      </p>
      <div className="space-y-2 text-left text-sm text-gray-600">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
          Fetching business data from OpenStreetMap
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
          Performing clustering analysis
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
          Identifying market opportunities
        </div>
      </div>
    </div>
  </div>
);

// Error Component
const ErrorScreen = ({ error, onRetry }) => (
  <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center px-4">
    <div className="bg-white rounded-2xl shadow-xl p-12 text-center max-w-md w-full">
      <AlertCircle className="h-16 w-16 text-red-600 mx-auto mb-6" />
      <h2 className="text-2xl font-bold text-gray-900 mb-2">Analysis Failed</h2>
      <p className="text-gray-600 mb-6">{error}</p>
      <button
        onClick={onRetry}
        className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors font-medium"
      >
        Try Again
      </button>
    </div>
  </div>
);

// Main App Component
const ShopTypeAnalyzer = () => {
  const [view, setView] = useState('input'); // 'input', 'loading', 'results', 'error'
  const [city, setCity] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  const analyzeCity = async (cityName) => {
    setCity(cityName);
    setView('loading');
    setError('');

    try {
      const response = await fetch('http://localhost:5000/api/business', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ city: cityName }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to analyze city');
      }

      const data = await response.json();
      
      if (!data || data.length === 0) {
        throw new Error('No business opportunities found for this city');
      }

      setResults(data);
      setView('results');
    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'Failed to analyze city. Please try again.');
      setView('error');
    }
  };

  const handleNewSearch = () => {
    setView('input');
    setCity('');
    setResults(null);
    setError('');
  };

  const handleBackToSelection = () => {
    // Navigate back to selection page - you can customize this
    window.history.back(); // Or use your router's navigation
  };

  const handleRetry = () => {
    if (city) {
      analyzeCity(city);
    } else {
      setView('input');
    }
  };

  return (
    <>
      <style>{`
        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }
      `}</style>

      {view === 'input' && <FindShopType onAnalyze={analyzeCity} onBack={handleBackToSelection} />}
      {view === 'loading' && <LoadingScreen city={city} />}
      {view === 'results' && (
        <ShopTypeResults
          city={city}
          results={results}
          onNewSearch={handleNewSearch}
        />
      )}
      {view === 'error' && <ErrorScreen error={error} onRetry={handleRetry} />}
    </>
  );
};

export default ShopTypeAnalyzer;