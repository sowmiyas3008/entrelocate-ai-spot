
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



