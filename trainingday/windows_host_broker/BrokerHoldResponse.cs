using System;

namespace Vera.TrainingDay.HostBroker {
  // A diagnostic response, NEVER an authorization token. The installed broker
  // has no owner-selected Tier 1 policy or independent Tier 2 witnesses.
  internal static class BrokerHoldResponse {
    internal static string Format(string request, bool authenticated, int servicePid,
                                  string bootId) {
      if (servicePid <= 0 || String.IsNullOrEmpty(bootId))
        throw new ArgumentException("invalid host-owned diagnostic identity");
      string reason = !authenticated ? "CLIENT_TOKEN_NOT_ADMIN" :
        request == "STATUS_V1" ? "HOST_COMPONENT_PRESENT_BUT_UNQUALIFIED" :
        request == "PREPARE_TIER1_V1" ? "TIER1_OWNER_POLICY_UNDECIDED" :
        "NO_INDEPENDENT_GPU_ROUTE_RIGHTS_WITNESS";
      // Never reflect caller-supplied request bytes or claim grant fields.
      return "{\"schema\":\"VERA_HOST_BROKER_STATUS_V1\",\"state\":\"HOLD\","+
        "\"reason\":\""+reason+"\",\"client_authenticated\":"+(authenticated?"true":"false")+","+
        "\"service_pid\":"+servicePid+","+
        "\"service_boot_id\":\""+bootId+"\","+
        "\"authorized_launches\":0,\"gpu_lease_issued\":false,"+
        "\"data_rights_verified\":false,\"nas_witness_verified\":false,\"training_observed\":false}";
    }
  }
}
