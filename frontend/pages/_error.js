export default function ErrorPage({ statusCode }) {
  return (
    <div
      style={{
        padding: 40,
        fontFamily: "Arial, sans-serif",
        color: "#e6eef8",
        background: "#0f1724",
        minHeight: "100vh",
      }}
    >
      <h1>Application error</h1>
      <p>
        {statusCode
          ? `An error ${statusCode} occurred on the server.`
          : "An error occurred on the client."}
      </p>
      <p>Please refresh the page.</p>
    </div>
  );
}

ErrorPage.getInitialProps = ({ res, err }) => {
  const statusCode = res?.statusCode || err?.statusCode || 500;
  return { statusCode };
};
