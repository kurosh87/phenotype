"use client";

import React, { useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";

interface GeoJSONFeature {
  type: "Feature";
  geometry: { type: "Polygon"; coordinates: number[][][] };
  properties: {
    intensity: "primary" | "secondary";
    color: string;
    area_pixels: number;
  };
}

interface GeoJSONFeatureCollection {
  type: "FeatureCollection";
  features: GeoJSONFeature[];
}

interface PhenotypeDistributionMapProps {
  geojson: GeoJSONFeatureCollection | null;
  phenotypeName: string;
  className?: string;
}

export default function PhenotypeDistributionMap({
  geojson,
  phenotypeName,
  className = "",
}: PhenotypeDistributionMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);

  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    // Initialize Mapbox (using default public token for development)
    mapboxgl.accessToken =
      process.env.NEXT_PUBLIC_MAPBOX_TOKEN ||
      "pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw";

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/light-v11",
      center: [0, 30],
      zoom: 1.5,
      projection: { name: "naturalEarth" } as any,
    });

    map.current.on("load", () => {
      setMapLoaded(true);
    });

    return () => {
      map.current?.remove();
      map.current = null;
    };
  }, []);

  useEffect(() => {
    if (!map.current || !mapLoaded || !geojson) return;

    // Remove existing layers and sources
    if (map.current.getLayer("distribution-primary")) {
      map.current.removeLayer("distribution-primary");
    }
    if (map.current.getLayer("distribution-secondary")) {
      map.current.removeLayer("distribution-secondary");
    }
    if (map.current.getLayer("distribution-primary-outline")) {
      map.current.removeLayer("distribution-primary-outline");
    }
    if (map.current.getLayer("distribution-secondary-outline")) {
      map.current.removeLayer("distribution-secondary-outline");
    }
    if (map.current.getSource("distribution")) {
      map.current.removeSource("distribution");
    }

    // Add source
    map.current.addSource("distribution", {
      type: "geojson",
      data: geojson as any,
    });

    // Add secondary distribution layer (underneath)
    map.current.addLayer({
      id: "distribution-secondary",
      type: "fill",
      source: "distribution",
      filter: ["==", ["get", "intensity"], "secondary"],
      paint: {
        "fill-color": "#7F7F00",
        "fill-opacity": 0.4,
      },
    });

    // Add secondary outline
    map.current.addLayer({
      id: "distribution-secondary-outline",
      type: "line",
      source: "distribution",
      filter: ["==", ["get", "intensity"], "secondary"],
      paint: {
        "line-color": "#5F5F00",
        "line-width": 1,
        "line-opacity": 0.6,
      },
    });

    // Add primary distribution layer (on top)
    map.current.addLayer({
      id: "distribution-primary",
      type: "fill",
      source: "distribution",
      filter: ["==", ["get", "intensity"], "primary"],
      paint: {
        "fill-color": "#FFFF00",
        "fill-opacity": 0.6,
      },
    });

    // Add primary outline
    map.current.addLayer({
      id: "distribution-primary-outline",
      type: "line",
      source: "distribution",
      filter: ["==", ["get", "intensity"], "primary"],
      paint: {
        "line-color": "#CCCC00",
        "line-width": 1.5,
        "line-opacity": 0.8,
      },
    });

    // Calculate bounds to fit all features
    if (geojson.features.length > 0) {
      const bounds = new mapboxgl.LngLatBounds();

      geojson.features.forEach((feature) => {
        feature.geometry.coordinates[0].forEach((coord) => {
          bounds.extend(coord as [number, number]);
        });
      });

      map.current.fitBounds(bounds, {
        padding: 40,
        maxZoom: 5,
      });
    }

    // Add popup on hover
    const popup = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false,
    });

    const onMouseMove = (e: mapboxgl.MapLayerMouseEvent) => {
      if (!e.features || e.features.length === 0) return;

      map.current!.getCanvas().style.cursor = "pointer";

      const feature = e.features[0];
      const intensity = feature.properties?.intensity || "unknown";
      const areaPixels = feature.properties?.area_pixels || 0;

      popup
        .setLngLat(e.lngLat)
        .setHTML(
          `<div style="padding: 4px 8px;">
            <strong>${phenotypeName}</strong><br/>
            <span style="text-transform: capitalize;">${intensity}</span> distribution<br/>
            <span style="font-size: 0.85em; color: #666;">Area: ${Math.round(areaPixels)} px²</span>
          </div>`
        )
        .addTo(map.current!);
    };

    const onMouseLeave = () => {
      map.current!.getCanvas().style.cursor = "";
      popup.remove();
    };

    map.current.on("mousemove", "distribution-primary", onMouseMove);
    map.current.on("mousemove", "distribution-secondary", onMouseMove);
    map.current.on("mouseleave", "distribution-primary", onMouseLeave);
    map.current.on("mouseleave", "distribution-secondary", onMouseLeave);

    return () => {
      map.current?.off("mousemove", "distribution-primary", onMouseMove);
      map.current?.off("mousemove", "distribution-secondary", onMouseMove);
      map.current?.off("mouseleave", "distribution-primary", onMouseLeave);
      map.current?.off("mouseleave", "distribution-secondary", onMouseLeave);
    };
  }, [mapLoaded, geojson, phenotypeName]);

  if (!geojson || geojson.features.length === 0) {
    return (
      <div
        className={`flex items-center justify-center bg-gray-100 rounded-lg ${className}`}
      >
        <p className="text-gray-500">No distribution data available</p>
      </div>
    );
  }

  const primaryCount = geojson.features.filter(
    (f) => f.properties.intensity === "primary"
  ).length;
  const secondaryCount = geojson.features.filter(
    (f) => f.properties.intensity === "secondary"
  ).length;

  return (
    <div className={`relative ${className}`}>
      <div ref={mapContainer} className="w-full h-full rounded-lg" />
      <div className="absolute bottom-4 right-4 bg-white/90 backdrop-blur-sm px-3 py-2 rounded-lg shadow-lg text-sm">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 bg-yellow-400 border border-yellow-600 rounded"></div>
            <span>Primary ({primaryCount})</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 bg-yellow-700/60 border border-yellow-800 rounded"></div>
            <span>Secondary ({secondaryCount})</span>
          </div>
        </div>
      </div>
    </div>
  );
}
