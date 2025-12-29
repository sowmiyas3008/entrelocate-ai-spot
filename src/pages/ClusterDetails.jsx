// import { useLocation, useNavigate } from "react-router-dom";
// import { Button } from "@/components/ui/button";
// import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
// import "leaflet/dist/leaflet.css";

// const ClusterDetails = () => {
//   const { state } = useLocation();
//   const navigate = useNavigate();
//   const { city, shopCategory, cluster } = state || {};

//   if (!cluster) return <p>No cluster data</p>;

//   return (
//     <div className="min-h-screen p-6">
//       <h1 className="text-3xl font-bold mb-2">
//         {cluster.cluster} – {shopCategory}
//       </h1>

//       <p className="mb-4 text-muted-foreground">
//         Recommended neighborhoods:
//       </p>

//       <ul className="mb-6 list-disc list-inside">
//         {cluster.neighborhood_areas.map((n, i) => (
//           <li key={i}>{n}</li>
//         ))}
//       </ul>

//       {/* Map */}
//       <MapContainer
//         center={[13.0827, 80.2707]} // temporary center (later dynamic)
//         zoom={13}
//         className="h-96 w-full rounded-lg"
//       >
//         <TileLayer
//           url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
//         />
//         <Marker position={[13.0827, 80.2707]}>
//           <Popup>{cluster.cluster}</Popup>
//         </Marker>
//       </MapContainer>

//       <Button className="mt-6" onClick={() => navigate(-1)}>
//         Back
//       </Button>
//     </div>
//   );
// };

// export default ClusterDetails;

// import { useLocation, useNavigate } from "react-router-dom";
// import { Button } from "@/components/ui/button";
// import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
// import "leaflet/dist/leaflet.css";

// const ClusterDetails = () => {
//   const { state } = useLocation();
//   const navigate = useNavigate();
//   const { city, shopCategory, cluster } = state || {};

//   if (!cluster) {
//     return <p className="p-6">No cluster data available</p>;
//   }

//   /* ===============================
//      ✅ DYNAMIC MAP CENTER
//      =============================== */
//   const latitude =
//     cluster.latitude ??
//     cluster.lat ??
//     cluster.center?.lat;

//   const longitude =
//     cluster.longitude ??
//     cluster.lng ??
//     cluster.center?.lng;

//   // Safety check
//   if (!latitude || !longitude) {
//     return (
//       <p className="p-6 text-red-500">
//         Cluster location coordinates missing
//       </p>
//     );
//   }

//   const mapCenter = [latitude, longitude];

//   /* ===============================
//      ✅ NEIGHBORHOOD VISIBILITY
//      =============================== */
//   const validNeighborhoods =
//     Array.isArray(cluster.neighborhood_areas) &&
//     cluster.neighborhood_areas.length > 0 &&
//     !cluster.neighborhood_areas.includes("N/A");

//   return (
//     <div className="min-h-screen p-6 space-y-6">
//       {/* Header */}
//       <div>
//         <h1 className="text-3xl font-bold">
//           {cluster.cluster} – {shopCategory}
//         </h1>
//         <p className="text-muted-foreground">
//           Best location in {city}
//         </p>
//       </div>

//       {/* Neighborhoods */}
//       {validNeighborhoods && (
//         <div>
//           <p className="mb-2 text-muted-foreground">
//             Recommended neighborhoods:
//           </p>
//           <ul className="list-disc list-inside">
//             {cluster.neighborhood_areas.map((n, i) => (
//               <li key={i}>{n}</li>
//             ))}
//           </ul>
//         </div>
//       )}

//       {/* 🗺️ MAP */}
//       <MapContainer
//         center={mapCenter}
//         zoom={13}
//         className="h-96 w-full rounded-lg"
//       >
//         <TileLayer
//           url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
//           attribution="© OpenStreetMap contributors"
//         />

//         <Marker position={mapCenter}>
//           <Popup>
//             <strong>{cluster.cluster}</strong>
//             <br />
//             {shopCategory}
//           </Popup>
//         </Marker>
//       </MapContainer>

//       {/* Actions */}
//       <Button onClick={() => navigate(-1)}>
//         Back
//       </Button>
//     </div>
//   );
// };

// export default ClusterDetails;


import { useLocation, useNavigate } from "react-router-dom";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import { Button } from "@/components/ui/button";
import "leaflet/dist/leaflet.css";

const ClusterDetails = () => {
  const { state } = useLocation();
  const navigate = useNavigate();
  const { cluster, shopCategory } = state || {};

  if (!cluster || !cluster.latitude || !cluster.longitude) {
    return <p>Cluster location coordinates missing</p>;
  }

  const validNeighborhoods =
    cluster.neighborhood_areas?.filter(n => n && n !== "N/A") || [];

  return (
    <div className="min-h-screen p-6">
      <h1 className="text-3xl font-bold mb-4">
        {cluster.cluster} – {shopCategory}
      </h1>

      {validNeighborhoods.length > 0 && (
        <>
          <p className="mb-2 text-muted-foreground">Recommended neighborhoods:</p>
          <ul className="mb-6 list-disc list-inside">
            {validNeighborhoods.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
          <p className="text-sm text-muted-foreground">
            Estimated foot traffic: 
            <span className="font-semibold ml-1">
            {cluster.traffic_score}
            </span>
          </p>

        </>
      )}


      

      <MapContainer
        center={[cluster.latitude, cluster.longitude]}
        zoom={14}
        className="h-96 w-full rounded-lg"
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

        <Marker position={[cluster.latitude, cluster.longitude]}>
          <Popup>
            {cluster.cluster}<br />
            {cluster.places_count} places
          </Popup>
        </Marker>
      </MapContainer>

      <Button className="mt-6" onClick={() => navigate(-1)}>
        Back
      </Button>
    </div>
  );
};

export default ClusterDetails;



