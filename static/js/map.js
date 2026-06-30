var map = L.map("map")
.setView([16.5062,80.6480],7);

L.tileLayer(
"https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
{
maxZoom:19
}
).addTo(map);

L.marker(
[16.5062,80.6480]
).addTo(map)
.bindPopup(
"Disease Outbreak Area"
);
